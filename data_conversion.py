import pandas as pd
import re

# ---------------------------------------------------------
# 1. 定义分类与关键词映射 (基于之前的分析)
# key: category index (0-9)
# value: description
# ---------------------------------------------------------
taste_categories = {
    0: "Sweet & Fruity (甜味/果香)",
    1: "Bitter (苦味)",
    2: "Acidic (酸味)",
    3: "Salty & Mineral (咸味/矿物质)",
    4: "Soft & Smooth (软/滑)",
    5: "Hard, Rough & Dry (硬/涩/干)",
    6: "Stimulating & Carbonated (刺激/气泡)",
    7: "Fresh, Cool & Clean (清新/凉爽/自然)",
    8: "Tasteless & Neutral (无味/中性)",
    9: "Artificial & Off-flavor (人工味/异味)"
}

# 关键词匹配逻辑
# 格式: (Category Index, [Positive Keywords], [Negative Keywords to exclude])
# 注意：关键词均为小写
mapping_rules = [
    (0, ["sweet", "fruit", "aroma", "candy", "bright", "sugar", "tasty"], ["not sweet", "no lingering sweetness"]),
    (1, ["bitter"], []),
    (2, ["acid"], []),
    (3, ["salt", "mineral", "dust", "calcium", "iron", "metal"], []),
    (4, ["soft", "smooth", "gentle", "mild", "lubricate", "moist", "visc", "delicate"], []),
    (5, ["hard", "dry", "rough", "sticky", "boil"], ["not dry"]),
    (6, ["stimul", "thrill", "irrit", "gas", "spark", "tingle", "energ"], ["no stimul", "no thrill"]),
    (7, ["fresh", "cool", "clean", "crisp", "nature", "mountain", "pale", "summer"], ["not fresh", "not crisp"]),
    (8, ["tasteless", "neutral", "normal", "quench", "no material"], []),
    (9, ["chem", "city", "dirty", "many material"], [])
]

# ---------------------------------------------------------
# 2. 辅助函数：将文本描述分类为 Index 列表
# ---------------------------------------------------------
def classify_text(text):
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    found_categories = set()
    
    for cat_idx, pos_keywords, neg_keywords in mapping_rules:
        # 1. 检查是否存在否定词 (如果有否定词，跳过该分类)
        # 例如：如果是 "not sweet"，我们不希望它匹配到 sweet
        has_negative = any(neg in text for neg in neg_keywords)
        if has_negative:
            continue
            
        # 2. 检查是否有正面关键词
        if any(pos in text for pos in pos_keywords):
            found_categories.add(cat_idx)
            
    return list(found_categories)

# ---------------------------------------------------------
# 3. 主处理逻辑
# ---------------------------------------------------------
def process_data(input_file, output_file):
    try:
        # 读取 CSV
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: 找不到文件 {input_file}")
        return

    # 获取水的种类 (第一列是 Respondent Details，所以从第二列开始)
    water_columns = df.columns[1:]
    water_types = {idx: name for idx, name in enumerate(water_columns)}
    
    output_data = []

    # 遍历每一列 (每种水)
    for col_idx, water_name in enumerate(water_columns):
        water_type_idx = col_idx  # 0-based index for water
        
        # 遍历该列下的每一个单元格 (每一个 Respondent 的描述)
        for cell_content in df[water_name]:
            # 清理换行符，将多行内容合并以便搜索
            clean_text = str(cell_content).replace('\n', ' ').replace('\r', ' ')
            
            # 获取该描述对应的分类 Indices
            taste_indices = classify_text(clean_text)
            
            # 将每一个分类作为单独的数据点添加
            if taste_indices:
                for t_idx in taste_indices:
                    output_data.append([water_type_idx, t_idx])
            else:
                # 可选：如果无法分类，可以略过或标记为特定分类
                pass

    # ---------------------------------------------------------
    # 4. 输出结果
    # ---------------------------------------------------------
    
    # 打印 Mapping
    print("="*40)
    print("MAPPING REFERENCE (映射表)")
    print("="*40)
    
    print("\n[第一个集合] Water Types (0 - 7):")
    for idx, name in water_types.items():
        # 清理一下换行符让打印也好看点
        print(f"  {idx}: {name.replace(chr(10), ' ')}") 
        
    print("\n[第二个集合] Taste Categories (0 - 9):")
    for idx, desc in taste_categories.items():
        print(f"  {idx}: {desc}")
        
    print("="*40)

    # 保存为 CSV
    result_df = pd.DataFrame(output_data, columns=['water_type_idx', 'taste_idx'])
    result_df.to_csv(output_file, index=False)
    print(f"\nSuccess! 数据已转换并保存至: {output_file}")
    print(f"Total data points generated: {len(result_df)}")

# ---------------------------------------------------------
# 执行脚本
# ---------------------------------------------------------
if __name__ == "__main__":
    # 假设您的文件名为 waterdata.csv
    process_data('water_raw_data.csv', 'water_taste_data.csv')
