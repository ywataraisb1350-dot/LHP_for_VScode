import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def create_comparison_graph():
    # CSVファイルのパス
    csv_path = r"C:\Users\渡会侑雅\LHPcodes_lap\LHP_for_VScode\for_graph\KEYresult_ll3-4_5-8_compere.csv"
    
    # --- 1. データの読み込み ---
    # グループ1: 1行目がヘッダー、2〜7行目がデータ (6行分)
    df1 = pd.read_csv(csv_path, nrows=6)
    label1 = df1.columns[7] if len(df1.columns) > 7 else "Group 1"
    
    # グループ2: 9行目がヘッダー、10〜15行目がデータ (6行分)
    df2 = pd.read_csv(csv_path, skiprows=8, nrows=6)
    label2 = df2.columns[7] if len(df2.columns) > 7 else "Group 2"
    
    # --- 2. グラフの準備 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # A列(X軸の値)を取得
    x_vals = df1.iloc[:, 0].values
    
    # ① 各グループの棒グラフの間隔を少し開ける
    bar_width = 0.35  # 棒の幅
    gap = 0.05        # 隙間の幅
    x1_pos = x_vals - (bar_width + gap) / 2
    x2_pos = x_vals + (bar_width + gap) / 2
    
    # B列からF列までの列名を取得
    bar_columns = df1.columns[1:6]
    
    # ② 棒グラフの色を両グループで統一 (青、赤、オレンジ、緑、青)
    colors = ['magenta', 'red', 'orange', 'green', 'blue']
    
    # --- 3. 積み上げ棒グラフの描画 ---
    bottom1 = np.zeros(len(df1))
    bottom2 = np.zeros(len(df2))
    
    for i, col in enumerate(bar_columns):
        # グループ1
        ax.bar(x1_pos, df1[col], width=bar_width, bottom=bottom1, 
               color=colors[i], edgecolor='black', linewidth=0.5)
        bottom1 += df1[col].fillna(0).values
        
        # グループ2
        ax.bar(x2_pos, df2[col], width=bar_width, bottom=bottom2, 
               color=colors[i], edgecolor='black', linewidth=0.5)
        bottom2 += df2[col].fillna(0).values

    # --- 4. 折れ線グラフの描画 (G列) ---
    line_col_1 = df1.columns[6]
    line_col_2 = df2.columns[6]
    
    # ③ cap.について、色は黒で、実線＆○、破線＆× で描画してグループを区別
    line1, = ax.plot(x_vals, df1.iloc[:, 6], marker='o', color='black', 
            linewidth=2, markersize=8, linestyle='-', 
            label=f"{label1}")
    
    line2, = ax.plot(x_vals, df2.iloc[:, 6], marker='x', color='black', 
            linewidth=2, markersize=8, linestyle='--', 
            label=f"{label2}")

    # --- 5. グラフの装飾 ---
    ax.set_xlabel(df1.columns[0], fontsize=12)
    ax.set_ylabel("pressure [Pa]", fontsize=12)
    #ax.set_title("Comparison Graph", fontsize=14)
    ax.set_xticks(x_vals)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # ④ 凡例を2つに分ける
    # 凡例1: 折れ線（グループの違い）
    legend1 = ax.legend(handles=[line1, line2], loc='upper left', 
                        bbox_to_anchor=(1.02, 1), title="Cap. (Lines)")
    ax.add_artist(legend1) # 1つ目の凡例をグラフに追加（上書きを防ぐため）
    
    # 凡例2: 棒グラフ（色の違い）
    bar_patches = [mpatches.Patch(color=colors[i], edgecolor='black', label=col) for i, col in enumerate(bar_columns)]
    ax.legend(handles=bar_patches, loc='upper left', 
              bbox_to_anchor=(1.02, 0.8), title="Bar Components")

    # ⑤ x軸が最小の棒グラフ上部に、それぞれのグループ名を示す
    min_idx = np.argmin(x_vals) # x軸が最小になるインデックス
    x_min_1 = x1_pos[min_idx]
    x_min_2 = x2_pos[min_idx]
    y_min_1 = bottom1[min_idx]  # グループ1の一番左の棒のトータルの高さ
    y_min_2 = bottom2[min_idx]  # グループ2の一番左の棒のトータルの高さ
    
    # 棒グラフのすぐ上に文字を配置するためのオフセット（Y軸最大値の約3%の高さ）
    y_offset = max(max(bottom1), max(bottom2)) * 0.03
    
    # 線グラフと文字が被っても読みやすいように背景を白で薄く塗りつぶす設定(bbox)を入れています
    bbox_props = dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    ax.text(x_min_1, y_min_1 + y_offset, label1, ha='center', va='bottom', 
            fontsize=11, fontweight='bold', bbox=bbox_props)
    ax.text(x_min_2, y_min_2 + y_offset, label2, ha='center', va='bottom', 
            fontsize=11, fontweight='bold', bbox=bbox_props)

    # レイアウトを調整して表示（右側の凡例が見切れないようにする）
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_comparison_graph()