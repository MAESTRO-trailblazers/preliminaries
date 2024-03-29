import spacy
nlp = spacy.load("en_core_web_sm") # 加载 SpaCy的英文模型 en_core_web_sm(中文模型 zh_core_web_sm)
import os
from collections import Counter

def read_file(file_path): # 读取文本文件
    with open(file_path, 'r', encoding='utf-8') as file: # encoding是打开文件时所使用的编码格式，具体用什么要修改
        text = file.read()
    return text

def average_sentence_length(doc): # 计算平均句长
    num_sentences = len(list(doc.sents)) # 计算句子数量
    num_words = len([token for token in doc if not token.is_punct]) # 计算单词数量(不含标点)
    return num_words / num_sentences if num_sentences else 0

def parse_tree_depth(doc): # 计算解析树深度
    depths = []
    for token in doc:
        depth = 0
        current_token = token
        while current_token.head is not None and current_token.head != current_token:
            depth += 1
            current_token = current_token.head
        depths.append(depth)
    return max(depths) if depths else 0

def num_subordinate_clause(doc): # 计算从句数量
    num_clauses = 0
    for token in doc:
        if token.dep_ == "advcl" or token.dep_ == "acl":
            num_clauses += 1
    return num_clauses

def part_of_speech(doc): # 计算词性多样性
    pos_tags = [token.pos_ for token in doc] # 获取每个词元的词性标记
    pos_counts = Counter(pos_tags) # 统计每种词性标记的数量
    total_pos = len(pos_tags)
    diversity_score = sum((count / total_pos) ** 2 for count in pos_counts.values())
    return 1 - diversity_score # 值越接近1，词性越多样

def dependency_distance(doc): # 计算依存距离
    distances = {} # 存储依存距离的空字典
    for token in doc:
        if token.dep_ != "ROOT": # 排除根节点
            governor = token.head.i + 1  # 获取依存关系的支配词索引(SpaCy中的索引从 0开始，按习惯选择是否加 1)
            dependent = token.i + 1 # 获取依存关系的从属词索引
            distance = abs(governor - dependent) # 计算依存距离(支配词与从属词的索引之差的绝对值)
            if governor not in distances:
                distances[governor] = distance # 支配词索引是键，依存距离是值
            else:
                distances[governor] = max(distances[governor], distance) # 比较并更新，取较大值
    return max(distances.values())

def evaluate_text_level(file_paths): # 分级
    results = []
    for file_path in file_paths:
        text = read_file(file_path) # 读取文本文件
        doc = nlp(text) # 获得 text经过处理后的所有信息
        asl = average_sentence_length(doc) # 计算平均句长
        ptd = parse_tree_depth(doc) # 计算解析树深度
        nsc = num_subordinate_clause(doc) # 计算从句数量
        pos = part_of_speech(doc) # 计算词性多样性
        dd = dependency_distance(doc) # 计算依存距离

        print(f"{file_path}:")
        print(f"平均句长：{asl}")
        print(f"解析树深度：{ptd}")
        print(f"从句数量：{nsc}")
        print(f"词性多样性：{pos}")
        print(f"依存距离：{dd}\n")

        scores = {
            "average_sentence_length": asl,
            "parse_tree_depth": ptd,
            "num_subordinate_clauses": nsc,
            "part_of_speech": pos,
            "dependency_distance": dd
        }

        level_counts = {
            1: 0,
            2: 0,
            3: 0,
            4: 0
        }

        for score in scores.values(): # 这个要看具体以什么标准来分
            if score <= 10:
                level_counts[1] += 1
            elif score <= 20:
                level_counts[2] += 1
            elif score <= 30:
                level_counts[3] += 1
            else:
                level_counts[4] += 1
        
        max_level = max(level_counts, key=level_counts.get) # 找到字典中值最大的键(数量最多的级别)
        result = f"The text in {file_path} is classified as level {max_level}"
        results.append(result)

    return results

def batch_evaluate_text_level(folder_path): # 多篇文章分级
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"): # 以 .txt结尾的文本文件
                file_paths.append(os.path.relpath(os.path.join(root, file))) # 相对路径

    results = evaluate_text_level(file_paths)
    for result in results:
        print(result)

folder_path = r"/Users/mayiran/preliminaries/texts_to_be_rated"  # 替换为实际文件夹路径
batch_evaluate_text_level(folder_path)
