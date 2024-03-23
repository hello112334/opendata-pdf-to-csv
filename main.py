import pandas as pd
import tabula
import os
from constants import PREFECTURES


def delete_headers(df, line_number):
    if df.iloc[0, 0] == "基本情報" or (len(df.columns) > 1 and df.iloc[0, 1] == "基本情報"):
        return df.drop(df.index[:line_number])
    return df


# 大分県に不要なタイトルがあるため削除
def delete_title(df):
    if df.iloc[0, 0] == "緊急避妊に係る診療が可能な産婦人科医療機関等一覧":
        return df.drop(df.index[:1])
    return df


def fix_format_page_df(df, line_number):
    return delete_headers(delete_title(df), line_number)


if not os.path.exists("./output_files"):
    os.mkdir("./output_files")

for i, prefecture in enumerate(PREFECTURES, 1):
    print("PREFECTURE_NUMBER", i, prefecture)
    opendata_file = os.listdir(f"./data_files/shinryoujo_{i}")
    dfs = tabula.read_pdf(f"./data_files/shinryoujo_{i}/{opendata_file[0]}", lattice=True, pages='all', pandas_options={'header': None})
    # 1ページ目のみ「基本情報」行の削除のため1行指定
    first_df = fix_format_page_df(dfs[0], 1)
    # 2ページ目以降は「基本情報」およびヘッダーを削除するため2行指定
    dfs = [fix_format_page_df(x, 2) for x in dfs[1:]]
    dfs.insert(0, first_df)
    # ページごとのデータを結合
    df = pd.concat(dfs)
    # 改行コードを削除
    df = df.replace('\n', '').replace('\r', '').replace('\r\n', '').replace('\n\r', '')
    # データが2つ未満の行は不要な可能性が高いので行を削除 & 列名に欠損値がある場合も列ごと削除
    result_df = df.dropna(thresh=2).dropna(subset=[df.index[0]], axis=1)

    prefecture_number = str(i).zfill(2)
    result_df.to_csv(f"./output_files/{prefecture_number}_{prefecture}.csv", header=False, index=False)
