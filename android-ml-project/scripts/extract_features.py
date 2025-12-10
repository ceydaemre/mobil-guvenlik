import os
import re
import math
from typing import Optional, Dict
from androguard.misc import AnalyzeAPK
from tqdm import tqdm
import pandas as pd

# === PERMISSION SINIFLANDIRMALARI ===
DANGEROUS_PERMISSIONS = {
    "android.permission.READ_CONTACTS",
    "android.permission.WRITE_CONTACTS",
    "android.permission.GET_ACCOUNTS",
    "android.permission.READ_PHONE_STATE",
    "android.permission.CALL_PHONE",
    "android.permission.CAMERA",
    "android.permission.RECORD_AUDIO",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.ACCESS_COARSE_LOCATION",
    "android.permission.READ_SMS",
    "android.permission.SEND_SMS",
    "android.permission.RECEIVE_SMS",
    "android.permission.RECEIVE_MMS",
    "android.permission.READ_EXTERNAL_STORAGE",
    "android.permission.WRITE_EXTERNAL_STORAGE",
}

SIGNATURE_PERMISSIONS = {
    "android.permission.BIND_ACCESSIBILITY_SERVICE",
    "android.permission.BIND_AUTOFILL_SERVICE",
    "android.permission.BIND_VPN_SERVICE",
    "android.permission.BIND_DEVICE_ADMIN",
    "android.permission.BIND_NOTIFICATION_LISTENER_SERVICE",
}

# === OPCODE’LAR ===
IMPORTANT_OPCODES = {
    "invoke-virtual",
    "invoke-static",
    "const-string",
    "new-instance",
    "move",
    "goto",
}

# === API SIGNATURE’LAR ===
API_SIGNATURES = {
    "getDeviceId": "api_getDeviceId",
    "getSubscriberId": "api_getSubscriberId",
    "getLine1Number": "api_getLine1Number",
    "sendTextMessage": "api_sendTextMessage",
    "exec": "api_exec",
    "getRuntime": "api_runtime_getRuntime",
    "openConnection": "api_openConnection",
    "Socket": "api_newSocket",
    "Cipher.getInstance": "api_cipher_getInstance",
    "SecretKeySpec": "api_secretKeySpec_init",
    "Class.forName": "api_class_forName",
    "getDeclaredMethod": "api_getDeclaredMethod",
    "invoke": "api_method_invoke",
}

# === STRING ANALİZİ ===
def analyze_strings(a):
    strings = set()
    try:
        for s in a.get_strings():
            if isinstance(s, str):
                strings.add(s)
    except:
        pass

    url_count = sum(1 for x in strings if "http://" in x or "https://" in x)
    ip_count = sum(1 for x in strings if re.match(r"\d+\.\d+\.\d+\.\d+", x))
    base64_count = sum(1 for x in strings if re.match(r"[A-Za-z0-9+/]{20,}={0,2}$", x))
    hex_count = sum(1 for x in strings if re.match(r"^[0-9A-Fa-f]{20,}$", x))
    long_word_count = sum(1 for x in strings if len(x) > 40)

    return url_count, ip_count, base64_count, hex_count, long_word_count

# === PACKAGE NAME ENTROPY ===
def entropy(s: str) -> float:
    if not s:
        return 0.0
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(s)]
    return -sum(p * math.log(p, 2) for p in prob)

# === TEK APK ÖZELLİK ÇIKARIMI ===
def extract_features_from_apk(apk_path: str) -> Optional[Dict]:
    try:
        a, d, dx = AnalyzeAPK(apk_path)

        permissions = a.get_permissions()

        dangerous_count = sum(1 for p in permissions if p in DANGEROUS_PERMISSIONS)
        signature_count = sum(1 for p in permissions if p in SIGNATURE_PERMISSIONS)

        activities = len(a.get_activities())
        services = len(a.get_services())
        receivers = len(a.get_receivers())

        # Opcode sayımı
        opcode_counts = {f"op_{op}": 0 for op in IMPORTANT_OPCODES}

        # API call sayımı
        api_counts = {v: 0 for v in API_SIGNATURES.values()}

        for method in dx.get_methods():
            if method.is_external():
                continue
            for block in method.get_basic_blocks():
                for ins in block.get_instructions():
                    op = ins.get_name()
                    if op in IMPORTANT_OPCODES:
                        opcode_counts[f"op_{op}"] += 1

                    # API çağrısı tespiti
                    out = str(ins.get_output())
                    for api_key, api_name in API_SIGNATURES.items():
                        if api_key in out:
                            api_counts[api_name] += 1

        # String analizi
        url_c, ip_c, b64_c, hex_c, long_c = analyze_strings(a)

        # Native library var mı?
        has_native_lib = 1 if any(f.endswith(".so") for f in a.get_files()) else 0

        # Package name entropy
        pkg = a.get_package()
        pkg_entropy = entropy(pkg)

        # APK size
        apk_size = os.path.getsize(apk_path)

        features = {
            "permissions_count": len(permissions),
            "dangerous_permissions": dangerous_count,
            "signature_permissions": signature_count,
            "package_entropy": pkg_entropy,
            "has_native_lib": has_native_lib,
            "apk_size": apk_size,
            "activities": activities,
            "services": services,
            "receivers": receivers,
            "apk_name": os.path.basename(apk_path),
        }

        features.update(opcode_counts)
        features.update(api_counts)

        return features

    except Exception as e:
        print(f"[HATA] {apk_path}: {e}")
        return None

# === DATASET İŞLEME ===
def process_dataset(path, label):
    rows = []
    files = [f for f in os.listdir(path) if f.endswith(".apk")]

    for f in tqdm(files, desc=f"{label} analiz ediliyor"):
        apk_path = os.path.join(path, f)
        feats = extract_features_from_apk(apk_path)
        if feats:
            feats["class"] = label
            rows.append(feats)
    return rows

# === MAIN ===
if __name__ == "__main__":
    benign = process_dataset("dataset/benign", "benign")
    malware = process_dataset("dataset/malware", "malware")

    df = pd.DataFrame(benign + malware)
    df.to_csv("output/features.csv", index=False)
    print("[OK] → output/features.csv oluşturuldu")
