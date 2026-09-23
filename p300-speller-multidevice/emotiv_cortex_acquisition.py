"""
emotiv_cortex_acquisition.py

Emotiv Cortex WebSocket client for streaming raw EEG from EPOC Flex into LSL.
Replaces eeg_serial_collect_16ch.py in the P300 speller pipeline.

Prerequisites:
    pip install websocket-client python-dotenv
    Emotiv Cortex app (or EmotivPRO) must be running — it hosts the WebSocket
    server at wss://localhost:6868.

The auth flow on first run:
    1. requestAccess  -> Cortex app shows a dialog, user clicks Allow
    2. authorize      -> get cortexToken
    3. queryHeadsets  -> find the EPOC Flex
    4. createSession  -> open a session with the headset
    5. subscribe eeg  -> start receiving EEG samples
"""

import json
import ssl
import threading

import websocket

CORTEX_URL = "wss://localhost:6868"

_BOOKKEEPING_COLS = {
    "COUNTER", "INTERPOLATED", "MARKER_HARDWARE",
    "MARKERS", "RAW_CQ", "BATTERY", "BATTERY_PERCENT",
    "FwBufferSize", "FwClockTime",
}


class CortexClient:
    """
    Minimal synchronous JSON-RPC 2.0 client over WebSocket.
    Subscription events (EEG data) are dispatched via _eeg_callback.
    """

    def __init__(self, url=CORTEX_URL):
        self._next_id = 0
        self._lock = threading.Lock()
        self._pending = {}          # id -> {"event": Event, "result": dict}
        self._eeg_callback = None
        self._connected = threading.Event()

        self._ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        t = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}},
            daemon=True,
        )
        t.start()
        if not self._connected.wait(timeout=10):
            raise ConnectionError(
                "Cannot reach Cortex at %s — is Emotiv Cortex / EmotivPRO running?" % url
            )

    def _on_open(self, ws):
        self._connected.set()

    def _on_message(self, ws, raw):
        msg = json.loads(raw)
        if "eeg" in msg:
            if self._eeg_callback:
                self._eeg_callback(msg["eeg"])
            return
        mid = msg.get("id")
        if mid is not None:
            with self._lock:
                slot = self._pending.get(mid)
            if slot:
                slot["result"] = msg
                slot["event"].set()

    def _on_error(self, ws, error):
        print(f"[cortex] WebSocket error: {error}")

    def _on_close(self, ws, code, msg):
        print(f"[cortex] WebSocket closed (code={code})")

    def call(self, method, params, timeout=15):
        """Send a JSON-RPC request; block until the response arrives."""
        with self._lock:
            self._next_id += 1
            mid = self._next_id
            slot = {"event": threading.Event(), "result": None}
            self._pending[mid] = slot

        self._ws.send(json.dumps({
            "jsonrpc": "2.0", "method": method,
            "params": params, "id": mid,
        }))

        if not slot["event"].wait(timeout=timeout):
            with self._lock:
                self._pending.pop(mid, None)
            raise TimeoutError(
                f"No response for {method!r} (id={mid}) within {timeout}s"
            )

        with self._lock:
            self._pending.pop(mid, None)

        result = slot["result"]
        if "error" in result:
            raise RuntimeError(
                f"Cortex API error on {method!r}: {result['error']}"
            )
        return result.get("result")

    def close(self):
        self._ws.close()


class CortexAcquisition(threading.Thread):
    """
    Authenticates with Emotiv Cortex, subscribes to the 'eeg' stream,
    and forwards samples to an LSL outlet.

    Lifecycle:
        acq = CortexAcquisition(client_id, client_secret, stop_event)
        acq.connect()          # auth + channel discovery (blocks ~2-3 s)
        # inspect acq.channel_names, then create the LSL outlet
        acq.set_outlet(outlet)
        acq.start()            # begins pushing samples
        ...
        stop_event.set()       # signals the thread to unsubscribe and exit
        acq.join()
    """

    def __init__(self, client_id, client_secret, stop_event, headset_id=None):
        super().__init__(daemon=True)
        self.client_id = client_id
        self.client_secret = client_secret
        self.stop_event = stop_event
        self.headset_id = headset_id   # None = auto-select first found

        self.channel_names = []        # electrode names, set by connect()
        self._cols = []                # full column list from Cortex
        self._outlet = None
        self._client = None
        self._token = None
        self._session_id = None

    def connect(self):
        """
        Run the full Cortex auth + session + EEG-subscribe flow.
        Populates self.channel_names with electrode labels.
        Call from the main thread before start().
        """
        client = CortexClient()
        self._client = client

        # 1. Request access — on first use the user must approve in EMOTIV Launcher
        result = client.call("requestAccess", {
            "clientId": self.client_id,
            "clientSecret": self.client_secret,
        })
        if not result.get("accessGranted", False):
            print("\n[cortex] App not yet approved in EMOTIV Launcher.")
            print("         -> Open EMOTIV Launcher -> click the notification or go to")
            print("            Settings -> My Apps -> find your app -> click Allow.")
            print("         Press Enter here once you have approved it...")
            input()
            # Retry after user approves
            result = client.call("requestAccess", {
                "clientId": self.client_id,
                "clientSecret": self.client_secret,
            })
            if not result.get("accessGranted", False):
                raise RuntimeError(
                    "Access still not granted. Check that you approved the correct "
                    f"Client ID ({self.client_id}) in EMOTIV Launcher."
                )
        print(f"[cortex] access granted")

        # 2. Authorize -> cortexToken
        result = client.call("authorize", {
            "clientId": self.client_id,
            "clientSecret": self.client_secret,
            "debit": 1,
        })
        self._token = result["cortexToken"]
        print(f"[cortex] authorized (token={self._token[:12]}...)")

        # 3. Find the headset
        headsets = client.call("queryHeadsets", {})
        if not headsets:
            raise RuntimeError(
                "No headset found — is the EPOC Flex USB dongle plugged in?"
            )
        if self.headset_id:
            hs = next((h for h in headsets if h["id"] == self.headset_id), None)
            if hs is None:
                ids = [h["id"] for h in headsets]
                raise RuntimeError(
                    f"Headset {self.headset_id!r} not found; available: {ids}"
                )
        else:
            hs = headsets[0]
        self.headset_id = hs["id"]
        print(f"[cortex] headset: {hs['id']}  status={hs.get('status', '?')}")

        # 4. Create session
        result = client.call("createSession", {
            "cortexToken": self._token,
            "headset": self.headset_id,
            "status": "active",
        })
        self._session_id = result["id"]
        print(f"[cortex] session: {self._session_id}")

        # 5. Subscribe to EEG — this also reveals the column layout
        result = client.call("subscribe", {
            "cortexToken": self._token,
            "session": self._session_id,
            "streams": ["eeg"],
        })
        success = result.get("success", [])
        eeg_entry = next(
            (s for s in success if s.get("streamName") == "eeg"), None
        )
        if eeg_entry is None:
            failures = result.get("failure", [])
            raise RuntimeError(
                f"EEG subscription failed: {failures}\n"
                "Raw EEG on EPOC Flex requires an EmotivPRO license. "
                "Verify you are signed into the Cortex app with the licensed account."
            )

        cols = eeg_entry["cols"]
        self._cols = cols
        self.channel_names = [
            c for c in cols
            if c not in _BOOKKEEPING_COLS and not c.startswith("CQ_")
        ]
        print(f"[cortex] {len(self.channel_names)} EEG channels: {self.channel_names}")

    def set_outlet(self, outlet):
        self._outlet = outlet

    def run(self):
        """Push EEG samples to the LSL outlet until stop_event is set."""
        eeg_indices = [self._cols.index(ch) for ch in self.channel_names]

        def on_eeg(data):
            sample = [float(data[i]) for i in eeg_indices]
            self._outlet.push_sample(sample)

        self._client._eeg_callback = on_eeg

        # Block here — the WebSocket thread drives on_eeg via callbacks
        self.stop_event.wait()

        # Unsubscribe and close session cleanly
        try:
            self._client.call("unsubscribe", {
                "cortexToken": self._token,
                "session": self._session_id,
                "streams": ["eeg"],
            })
            self._client.call("updateSession", {
                "cortexToken": self._token,
                "session": self._session_id,
                "status": "close",
            })
        except Exception as exc:
            print(f"[cortex] cleanup warning: {exc}")
        finally:
            self._client.close()
