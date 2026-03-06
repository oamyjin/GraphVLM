# tree.py
import os

def print_tree(root, prefix=""):
    # 列出当前目录下的所有项目并排序
    items = os.listdir(root)
    items.sort()

    for i, name in enumerate(items):
        path = os.path.join(root, name)
        # 判断是不是最后一个项目，用不同符号绘制“树枝”
        connector = "└── " if i == len(items) - 1 else "├── "

        # 打印当前项目名
        print(prefix + connector + name)

        # 如果还是文件夹，则递归
        if os.path.isdir(path):
            # 继续生成前缀(注意空格与竖线的差别)
            sub_prefix = prefix + ("    " if i == len(items) - 1 else "│   ")
            print_tree(path, sub_prefix)

if __name__ == "__main__":
    # 递归打印当前目录下所有文件和文件夹
    print_tree(".")

