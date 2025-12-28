from utils.nodes_and_arcs import *
import numpy as np
import pandas as pd
import geopandas as gpd
import geohash2

# ================= 1. 数据加载与预处理 (保持不变) =================
node_gdf = load_node_gdf("data/node_gdf.geojson")
arc_gdf = load_arc_gdf("data/arc_gdf.geojson")
# 重置索引，确保 sjoin 后索引对齐
arc_gdf = arc_gdf.reset_index(drop=True)

print("正在加载订单数据...")
order_df = pd.read_csv("data/dataset_testing_20240527_20240614.csv")

print("正在转换坐标...")


def geohash_to_pos(gh):
    try:
        lat, lon, _, _ = geohash2.decode_exactly(gh)
        return np.array([lat, lon])
    except:
        return np.array([np.nan, np.nan])


start_pos = np.stack(order_df['start_geohash'].apply(geohash_to_pos).values)
end_pos = np.stack(order_df['end_geohash'].apply(geohash_to_pos).values)

order_df['start_lat'] = start_pos[:, 0]
order_df['start_lon'] = start_pos[:, 1]
order_df['end_lat'] = end_pos[:, 0]
order_df['end_lon'] = end_pos[:, 1]

# 转换为 GeoDataFrame
order_start_gdf, order_end_gdf = df2gdf(order_df)

# ================= 2. 核心：批量打标签 (用分号连接) =================
print("正在进行空间匹配...")


def get_joined_ids(order_gdf, region_gdf, id_col_name):
    """
    通用函数：执行空间连接，并将重叠区域的ID用分号拼接
    """
    # 1. 空间连接 (Left Join)
    joined = gpd.sjoin(
        order_gdf,
        region_gdf[[id_col_name, 'geometry']],
        how='left',
        predicate='within'
    )

    # 2. 分组聚合字符串
    # 逻辑：排除空值 -> 转为字符串 -> 排序(可选，保证顺序一致) -> 用分号连接
    # 结果示例: "(1, 2);(3, 4)" 或者 "10;11"
    grouped_ids = joined.groupby(joined.index)[id_col_name].apply(
        lambda x: ";".join(sorted([str(i) for i in x if pd.notnull(i)]))
    )
    return grouped_ids


# --- A. 匹配起点所在的 Arc ---
order_df['start_arc_ids'] = get_joined_ids(order_start_gdf, arc_gdf, 'arc_id')

# --- B. 匹配终点所在的 Arc ---
order_df['end_arc_ids'] = get_joined_ids(order_end_gdf, arc_gdf, 'arc_id')

# --- C. 匹配起点/终点所在的 Node (如果Node也有重叠需求，也可以用这个) ---
# 假设 Node 也可能重叠，我们统一用这个逻辑，如果没有重叠，结果就是单个ID
order_df['start_node_ids'] = get_joined_ids(order_start_gdf, node_gdf, 'node_id')
order_df['end_node_ids'] = get_joined_ids(order_end_gdf, node_gdf, 'node_id')

# ================= 3. 查看结果与保存 =================
print("预览结果:")
print(order_df[['start_geohash', 'start_arc_ids', 'end_arc_ids']].head())

# 导出 Excel
output_path = "data/tagged_bike_data_semicolon.xlsx"
# 注意：现在 start_arc_ids 已经是字符串了，可以直接保存
order_df.to_excel(output_path, index=False)
print(f"处理完成，文件已保存至: {output_path}")