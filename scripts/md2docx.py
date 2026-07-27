"""
md2docx.py · 把带 Mermaid / 数学公式 / 图片的 Markdown 转成 Word(.docx)

- Mermaid 代码块 → 用 mmdc 渲染成 PNG 再嵌入（Word 不能渲染 Mermaid）
- $...$ / $$...$$ 数学 → Word 原生公式
- 表格 / 图片 → Word 原生

依赖：pip install pypandoc-binary ; npm i -g @mermaid-js/mermaid-cli
用法：python scripts/md2docx.py <输入.md> [输出.docx]
"""
import os, re, sys, subprocess

MMDC = r"C:\Users\Administrator\AppData\Roaming\npm\mmdc.cmd"


def render_mermaid(code, out_png, workdir):
    mmd = os.path.join(workdir, "_m.mmd")
    pp = os.path.join(workdir, "_pp.json")
    open(mmd, "w", encoding="utf-8").write(code)
    open(pp, "w", encoding="utf-8").write('{"args":["--no-sandbox"]}')
    cmd = f'"{MMDC}" -i "{mmd}" -o "{out_png}" -b white -p "{pp}" -s 2'
    subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return os.path.exists(out_png)


def main():
    inp = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.splitext(inp)[0] + ".docx"
    workdir = os.path.dirname(inp)
    md = open(inp, encoding="utf-8").read()

    blocks = list(re.finditer(r"```mermaid\s*\n(.*?)\n```", md, re.S))
    print(f"找到 {len(blocks)} 个 Mermaid 图，逐个渲染 ...")
    for i in range(len(blocks) - 1, -1, -1):          # 从后往前替换，保持位置
        b = blocks[i]
        png = f"_mermaid_{i}.png"
        ok = render_mermaid(b.group(1), os.path.join(workdir, png), workdir)
        rep = f"![流程图{i+1}]({png})" if ok else b.group(0)
        print(f"  图{i+1}: {'渲染OK' if ok else '失败(保留代码)'}")
        md = md[:b.start()] + rep + md[b.end():]

    tmp = os.path.join(workdir, "_for_docx.md")
    open(tmp, "w", encoding="utf-8").write(md)

    import pypandoc
    pypandoc.convert_file(tmp, "docx", format="markdown", outputfile=out,
                          extra_args=["--resource-path", workdir, "--wrap=none"])
    # 清理中间文件
    for f in ["_m.mmd", "_pp.json", "_for_docx.md"]:
        p = os.path.join(workdir, f)
        if os.path.exists(p): os.remove(p)
    print(f"[OK] 生成 {out}")


if __name__ == "__main__":
    main()
