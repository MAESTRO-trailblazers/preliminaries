import spacy
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm") # 加载 SpaCy的英文模型 en_core_web_sm(中文模型 zh_core_web_sm)

def read_file(file_path): # 读取文本文件
    with open(file_path, 'r', encoding='utf-8') as file: # encoding是打开文件时所使用的编码格式，具体用什么要修改
        text = file.read()
        print("The file has been read.")
    return text

def average_sentence_length(doc): # 计算平均句长
    num_sentences = len(list(doc.sents)) # 计算句子数量
    num_words = len([token for token in tqdm(doc, desc = 'Processing: Average Sentence Length') if not token.is_punct]) # 计算单词数量(不含标点)
    final = num_words / num_sentences if num_sentences else 0
    print("\nThe ASL is " + str(final))
    return final

def parse_tree_depth(doc): # 计算解析树深度
    depths = []
    for token in tqdm(doc, desc = 'Processing: Parse Tree Depth'):
        depth = 0
        current_token = token
        while current_token.head is not current_token:
            depth += 1
            current_token = current_token.head
        depths.append(depth)
    final = max(depths) if depths else 0
    print("The PST is " + str(final))
    return final

def num_subordinate_clause(doc): # 计算从句数量
    num_clauses = 0
    for token in tqdm(doc, desc = 'Processing: Subordinate Clauses'):
        if token.dep_ == "mark" and any(child.dep_ == "ccomp" or child.dep_ == "xcomp" for child in token.children):
            num_clauses += 1
    print("The NSC is " + str(num_clauses))
    return num_clauses

def part_of_speech(doc): # 计算词性多样性
    pos_tags = [token.pos_ for token in tqdm(doc, desc = 'Processing: POS diveristy')]
    final = len(set(pos_tags)) / len(pos_tags)
    print("The POSD is " + str(final))
    return final

def dependency_distance(doc): # 计算依存距离
    distances = {} # 存储依存距离的空字典
    for token in tqdm(doc, desc = 'Processing: Dependency Distance'):
        if token.dep_ != "ROOT": # 排除根节点
            governor = token.head.i + 1  # 获取依存关系的支配词索引(SpaCy中的索引从 0开始，按习惯选择是否加 1)
            dependent = token.i + 1 # 获取依存关系的从属词索引
            distance = abs(governor - dependent) # 计算依存距离(支配词与从属词的索引之差的绝对值)
            if governor not in distances:
                distances[governor] = distance # 支配词索引是键，依存距离是值
            else:
                distances[governor] = max(distances[governor], distance) # 比较并更新，取较大值
    final = max(distances.values())
    print("The DD is " + str(final))
    return final

def evaluate_text_level(file_path): # 分级
    text = read_file(file_path) # 读取文本文件
    doc = nlp(text) # 获得 text经过处理后的所有信息
    asl = average_sentence_length(doc) # 计算平均句长

    nsc = num_subordinate_clause(doc) # 计算从句数量
    pos = part_of_speech(doc) # 计算词性多样性
    dd = dependency_distance(doc) # 计算依存距离

    ptd = parse_tree_depth(doc)  # 计算解析树深度

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

    for score in tqdm(scores.values(), desc = 'Processing: Final Score'): # 这个要看具体以什么标准来分
        if score <= 10:
            level_counts[1] += 1
        elif score <= 20:
            level_counts[2] += 1
        elif score <= 30:
            level_counts[3] += 1
        else:
            level_counts[4] += 1

    max_level = max(level_counts, key=level_counts.get) # 找到字典中值最大的键(数量最多的级别)
    return f"The text is classified as level {max_level}"

file_path = "thinking_as_a_hobby.txt" # 替换为实际文件路径
result = evaluate_text_level(file_path)
print(result)
