import pandas as pd
import jieba
import re


def clean_text(text):
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"\[\S+\]", "", text)  # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)  # 去除网址
    text = text.replace("转发微博", "")  # 去除无意义的词语
    text = re.sub(r"\s+", " ", text)  # 合并正文中过多的空格
    del_word = ["O网页链接", "展开c"]  # 定义要删除的子串列表
    # 使用正则表达式删除子串
    for p in del_word:
        text = re.sub(p, "", text)

    return text.strip()


def run_jieba(text):
    jieba_out = jieba.lcut(text)
    return jieba_out


if __name__ == '__main__':
    rawdata_file = "BCI.xlsx"
    output_file = "clean_BCI.xlsx"

    # 读取数据
    rawdata = pd.read_excel(rawdata_file)

    # 删除重复
    rawdata.drop_duplicates(subset=["博主id", "博文"], inplace=True)

    # 删除空值
    rawdata.dropna(subset=["博文"], inplace=True)

    # 整理博文内容
    rawdata["干净博文"] = rawdata["博文"].apply(clean_text)
    rawdata["干净博文"] = rawdata["干净博文"].map(str.lower)  # 统一为小写字母

    # jieba分词
    # rawdata["jieba"] = rawdata["干净博文"].apply(run_jieba)
    rawdata.to_excel(output_file, index=False)
