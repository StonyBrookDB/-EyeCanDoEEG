"""
check_license.py

Diagnostic for Cortex API error -32022 ("Device limit on this license has
been reached"). Walks the auth flow one call at a time and prints everything
Cortex will tell us about the account/license state:

    requestAccess -> authorize (debit=0, so it doesn't spend a session)
                  -> getUserLogin      (which Emotiv ID(s) are logged into
                                         the LOCAL Cortex/EMOTIV Launcher --
                                         no token needed)
                  -> getLicenseInfo    (needs a cortexToken, so only runs if
                                         authorize actually succeeded)
                  -> getUserInformation

If authorize itself throws -32022, we can't get a cortexToken (that's the
whole problem), so getLicenseInfo can't be called in that case -- the script
says so explicitly instead of crashing, and getUserLogin still runs since it
needs no token.

Usage:
    python check_license.py
"""
import os

from dotenv import load_dotenv

from emotiv_cortex_acquisition import CortexClient

load_dotenv()

client_id = os.environ["EMOTIV_CLIENT_ID"]
client_secret = os.environ["EMOTIV_CLIENT_SECRET"]


def section(title):
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


client = CortexClient()

# 1. requestAccess
section("requestAccess")
result = client.call("requestAccess", {
    "clientId": client_id,
    "clientSecret": client_secret,
})
print(result)
if not result.get("accessGranted", False):
    print("\n[!] App not approved in EMOTIV Launcher yet.")
    print("    Open EMOTIV Launcher -> Settings -> My Apps -> Allow, then re-run.")
    raise SystemExit(1)

# 2. getUserLogin -- no token required, shows who Cortex thinks is logged in
#    on THIS machine right now.
section("getUserLogin (local machine, no token needed)")
try:
    logins = client.call("getUserLogin", {})
    print(logins)
    if not logins:
        print("[!] No Emotiv ID logged in via EMOTIV Launcher on this machine.")
except Exception as exc:
    print(f"[!] getUserLogin failed: {exc}")

# 3. authorize -- debit=0 so this doesn't consume a paid EEG session even if
#    it succeeds. This is also where -32022 is actually thrown.
section("authorize (debit=0)")
token = None
try:
    auth_result = client.call("authorize", {
        "clientId": client_id,
        "clientSecret": client_secret,
        "debit": 0,
    })
    token = auth_result["cortexToken"]
    print(f"[ok] got cortexToken: {token[:12]}...")
except RuntimeError as exc:
    print(f"[FAILED] {exc}")
    print(
        "\n[!] This is the same -32022 you hit before. authorize() is where "
        "Cortex enforces the license's device limit, and it fails BEFORE a "
        "cortexToken is issued -- so getLicenseInfo can't be called this run "
        "(it requires a token). Likely causes, most common first:\n"
        "  1. EmotivPRO or another Cortex-connected app is already holding an\n"
        "     authorized session for this same account (on this PC or another).\n"
        "     Fully quit EmotivPRO / other Cortex apps, then retry.\n"
        "  2. A previous run of this script (or a crash) left a session\n"
        "     authorized without logging out, and the token hasn't expired yet.\n"
        "     Restart the EMOTIV Launcher / Cortex service to clear it.\n"
        "  3. Your Emotiv account's license genuinely has other devices\n"
        "     registered against it (e.g. phone app, another computer) --\n"
        "     check https://www.emotiv.com/my-account/devices/ (or the\n"
        "     'Devices' section of your Emotiv account) and remove old ones,\n"
        "     or ask EMOTIV support to raise the limit / clear stale devices."
    )

# 4. getLicenseInfo -- only possible if authorize succeeded above.
section("getLicenseInfo")
if token:
    try:
        info = client.call("getLicenseInfo", {"cortexToken": token})
        print(info)
    except Exception as exc:
        print(f"[!] getLicenseInfo failed: {exc}")
else:
    print("[skipped] no cortexToken from authorize() -- see above.")

# 5. getUserInformation -- also needs a token.
section("getUserInformation")
if token:
    try:
        info = client.call("getUserInformation", {"cortexToken": token})
        print(info)
    except Exception as exc:
        print(f"[!] getUserInformation failed: {exc}")
else:
    print("[skipped] no cortexToken from authorize() -- see above.")
