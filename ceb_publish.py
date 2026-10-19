"""CEB-OPENCORE · 开源发布脚手架（卫星件 4 · GitHub 发表两原则 A藏配方/B送服务）

按 v1.1 原则落地双输出：
  dist/public/   -> 送服务：API 表面 + 文档 + demo harness（可独立跑，价值自证）
  dist/private/  -> 藏配方：调优常量与词表（不随公开包发布）
  ceb_release_manifest.json -> 两侧清单 +  withheld 项 + sha256 指纹

纪律：本器只生成本地发布物，不联网、不推送（push 属二级动作，须善伦明示）。
"""
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "dist")

PUBLIC_FILES = ["ceb_policy.py", "ceb_allocator.py", "ceb_simharness.py",
                "ceb_realdata.py", "ceb_llmverify.py", "README.md"]
PUBLIC_DIRS = ["examples", "tests"]
WITHHELD = [
    "政策词表调优参数（BRANCH/CONVERGE/FALSIFY 命中阈值与档位宽度映射的最终取值）",
    "评测实例种子与容量上限的业务化取值（--seed/--cap 的产品化配置）",
    "对账器 five_axis 自评表（内部工件，不对外）",
    "内部语料提取配置（真实勾选卡/memory 路径与文件名，不对外）",
    "audit_repro.py 复核复跑器与全部内部 JSON 结果（过程工件，不对外）",
]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build():
    pub = os.path.join(DIST, "public")
    prv = os.path.join(DIST, "private")
    os.makedirs(pub, exist_ok=True)
    os.makedirs(prv, exist_ok=True)

    copied = []
    for name in PUBLIC_FILES:
        src = os.path.join(HERE, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(pub, name))
            copied.append(name)
    for d in PUBLIC_DIRS:
        src = os.path.join(HERE, d)
        if os.path.isdir(src):
            dst = os.path.join(pub, d)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied.append(f"{d}/")

    # private 侧只放一个占位说明（真配方不落公开目录）
    with open(os.path.join(prv, "WITHHELD.md"), "w", encoding="utf-8") as f:
        f.write("# 藏配方清单（不随公开包发布）\n\n")
        for w in WITHHELD:
            f.write(f"- {w}\n")

    fingerprints = {}
    for n in copied:
        p = os.path.join(pub, n)
        if os.path.isfile(p):
            fingerprints[n] = _sha256(p)
        else:
            fingerprints[n] = f"dir:{sum(len(fs) for _, _, fs in os.walk(p))} files"

    manifest = {
        "name": "cot-entropy-budget", "version": "1.0.0",
        "principle": "A藏配方/B送服务（GitHub 发表两原则 v1.1）",
        "public_files": copied,
        "withheld": WITHHELD,
        "fingerprints": fingerprints,
        "push_policy": "本文件只生成发布物；推送为对外动作，须人工明示授权后执行",
    }
    mpath = os.path.join(HERE, "ceb_release_manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


if __name__ == "__main__":
    m = build()
    print(json.dumps(m, ensure_ascii=False, indent=2))
    print("[写出] dist/ + ceb_release_manifest.json")
