"""
list_headsets.py

Standalone Cortex API check: prints every headset Cortex currently sees,
with its real connection status, so you can pick the right --headset-id
when you have more than one Emotiv device paired.

Prerequisites: same as p300_speller_experiment_emotiv_cortex.py --
Emotiv Cortex app (or EmotivPRO) running, EMOTIV_CLIENT_ID/SECRET in .env.

Usage:
    python list_headsets.py
"""
import os

from dotenv import load_dotenv

from emotiv_cortex_acquisition import CortexClient

load_dotenv()

client_id = os.environ["EMOTIV_CLIENT_ID"]
client_secret = os.environ["EMOTIV_CLIENT_SECRET"]

client = CortexClient()

result = client.call("requestAccess", {
    "clientId": client_id,
    "clientSecret": client_secret,
})
if not result.get("accessGranted", False):
    print("\n[cortex] App not yet approved in EMOTIV Launcher.")
    print("         -> Open EMOTIV Launcher -> click the notification or go to")
    print("            Settings -> My Apps -> find your app -> click Allow.")
    print("         Press Enter here once you have approved it...")
    input()
    result = client.call("requestAccess", {
        "clientId": client_id,
        "clientSecret": client_secret,
    })
    if not result.get("accessGranted", False):
        raise SystemExit("Access still not granted.")

headsets = client.call("queryHeadsets", {})
if not headsets:
    print("[cortex] No headsets found by Cortex at all -- check the app's own "
          "device list, not just this script.")
else:
    print(f"[cortex] {len(headsets)} headset(s) visible to Cortex:\n")
    for h in headsets:
        print(f"  id     : {h.get('id')}")
        print(f"  status : {h.get('status')}")
        print(f"  dongle : {h.get('dongle')}")
        print(f"  connectedBy : {h.get('connectedBy')}")
        print()
    print("Pass the id of whichever one shows status='connected' as:")
    print("  --headset-id <that id>")
