import os, subprocess

def _kc(service: str, account: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out or None
    except subprocess.CalledProcessError:
        return None

def wrds_user() -> str:
    return os.getenv("WRDS_USER") or _kc("wrds.user", "default") or ""

def wrds_password(user: str | None = None) -> str:
    u = user or wrds_user()
    return os.getenv("WRDS_PASSWORD") or _kc("wrds.pass", u) or ""

def wrds_creds() -> tuple[str, str]:
    u = wrds_user()
    p = wrds_password(u)
    if not u or not p:
        raise RuntimeError("WRDS credentials not found in env or Keychain.")
    return u, p
