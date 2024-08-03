import pandas as pd
import pdfplumber
import os
import requests
import re
import xml.etree.ElementTree as ET
import sys
import codecs

def load_prefectures(csv_file):
    """
    都道府県番号と名前の対応辞書をCSVファイルから読み込む
    """
    df = pd.read_csv(csv_file, dtype={
                     'number': int, 'name': str, 'english_name': str})
    return {row['english_name']: row['name'] for _, row in df.iterrows()}, list(df['english_name'])


# 都道府県の辞書と英語名のリストを読み込む
prefectures, PREFECTURES = load_prefectures('./table/prefectures.csv')

# csv出力フォマード+転換リストを読み込む
output_format_list_df = pd.read_csv(
    './table/output_format.csv', header=0, dtype=str)


def get_prefecture_name(prefecture_english_name):
    """
    都道府県の英語名から日本語名を取得
    """
    return prefectures.get(prefecture_english_name, "")


# CSVファイルを読み込み、郵便番号をキー、市区町村名を値とする辞書を作成
address_df = pd.read_csv(
    './data_files/ken_all/utf_ken_all.csv', header=None, dtype=str)
postal_to_location = {row[2].strip(): (row[6], row[7])
                      for row in address_df.values}
postal_to_location_code = {row[2].strip(): row[0] for row in address_df.values}
address_to_location_code = {
    (row[6].strip(), row[7].strip()): row[0] for row in address_df.values}
PREFECTURE_POSTAL_PREFIX = {
    row[1].strip(): row[0] for row in address_df.values}


def address_to_coordinates(address):
    """
    住所から緯度経度を取得
    """
    if not address:
        return 0, 0

    base_url = "http://geocode.csis.u-tokyo.ac.jp/cgi-bin/simple_geocode.cgi?charset=UTF8&addr="
    url = base_url + requests.utils.quote(str(address))
    latitude, longitude = 0, 0
    response = requests.get(url)
    if response.status_code == 200:
        xml_content = response.text
        xml_content = xml_content.replace("\n", "")
        root = ET.fromstring(xml_content)

        # 小数点以下第6位まで取得
        longitude = round(float(root.findtext(".//candidate/longitude")), 6)
        latitude = round(float(root.findtext(".//candidate/latitude")), 6)

    return latitude, longitude


def split_japanese_address(address):
    """
    住所を都道府県、市区町村、それ以降に分割
    """
    if not address:
        return ["", ""]

    pattern = re.compile(
        r'(?:(?P<region>...??[都道府県]))?'  # 都道府県 (オプション)
        r'(?P<locality>.+?[市区町村湾島])'  # 市区町村など
        r'(?P<remainder>.*)'  # それ以降の住所
    )

    match = pattern.match(address)
    if match:
        result = match.groupdict()
        region = result['region'] if result['region'] else ""
        locality = result['locality'] if result['locality'] else ""

        return [region, locality]
    else:
        return ["", ""]


def postal2location(postal_code):
    """
    郵便番号から市区町村を取得
    """

    if pd.isna(postal_code):
        return "", ""

    postal_code = postal_code.replace("-", "")
    if postal_code in postal_to_location:
        return postal_to_location[postal_code]

    return "", ""


def postal2location_code(postal_code):
    """
    郵便番号から市区町村コードを取得
    """

    if pd.isna(postal_code):
        return ""

    postal_code = postal_code.replace("-", "")
    if postal_code in postal_to_location_code:
        tmp_code = postal_to_location_code[postal_code]
        tmp_code = f'{tmp_code:05}'  # 0埋めで5文字
        tmp_code = convert_five_to_six_digit_code(tmp_code)  # 5桁を6桁に変換
        return tmp_code

    return ""


def address2location_code(prefecture, location):
    """
    都道府県と市区町村から市区町村コードを取得
    """
    if not prefecture or not location:
        return ""

    if (prefecture, location) in address_to_location_code:
        tmp_code = address_to_location_code[(prefecture, location)]
        tmp_code = f'{tmp_code:05}'  # 0埋めで5文字
        tmp_code = convert_five_to_six_digit_code(tmp_code)  # 5桁を6桁に変換
        return tmp_code

    return ""


def delete_title(df):
    """
    大分県に不要なタイトルがあるため削除
    """
    if df.iloc[0, 0] == "緊急避妊に係る診療が可能な産婦人科医療機関等一覧":
        return df.drop(df.index[:1])
    clear_change_line(df)
    return df


def delete_headers(df, line_number):
    """
    ヘッダー行を削除
    """
    target_list = ["基本情報", "施設名", "医療機関名"]
    for target in target_list:
        if df.iloc[0, 0] == target or (len(df.columns) > 1 and df.iloc[0, 1] == target):
            df = df.drop(df.index[:line_number])
    return df


def fix_format_page_df(df, line_number):
    """
    ページごとのデータフレームのフォーマットを修正
    """
    clear_change_line(df)
    return delete_headers(delete_title(df), line_number)


def clear_change_line(df):
    """
    行処理
    """
    # 改行コードと"を削除
    df.replace('\n', '', regex=True).replace('\r', '', regex=True).replace(
        '\r\n', '', regex=True).replace('\n\r', '', regex=True)
    df.replace('"', '', regex=True, inplace=True)

    # 時間表記の「~」を「-」に変換
    df.replace('~', '-', regex=True, inplace=True)
    df.replace('〜', '-', regex=True, inplace=True)

    # データが2つ未満の行は不要な可能性が高いので行を削除 & 列名に欠損値がある場合も列ごと削除
    df.dropna(axis=0, thresh=2, inplace=True)

    return df


def get_first_page(first_table, prefecture_name):
    """
    最初のページのヘッダーとデータを取得し、必要に応じてヘッダーに「公表の希望の有無」を追加
    """
    row = 1

    if prefecture_name == "新潟県":
        row = 0

    headers = first_table[row]

    # ヘッダーが「基本情報」になっている場合があるので、次のページのヘッダーを取得
    if headers[0] == "基本情報":
        row += 1
        headers = first_table[row]
    headers = [header.replace('\n', '').replace(
        '\r', '') if header else '' for header in first_table[row]]

    # 沖縄だけヘッダーの最初欄に「公表の希望の有無」を入れる
    if prefecture_name == "沖縄県":
        headers[0] = "公表の希望の有無"

    data = first_table[row+1:]
    return headers, data

# usage: python main.py               # output csv files (default)
# usage: python main.py --output-json # output json files
def main(argv):
    if argv[0] == '--output-json':
        print("Exporting to JSON files...")
    else:
        print("Exporting to CSV files...")

def unify_column_names(df, prefecture_name):
    """
    フォーマットのカラムを変換/統一
    """
    if prefecture_name not in output_format_list_df.columns:
        print(
            f"Error: {prefecture_name} is not in output_format_list_df columns")
        return df

    # 変換マッピングを取得
    column_mapping = output_format_list_df[prefecture_name].dropna().to_dict()

    # カラム名を変換
    # 例: 医療機関における緊急避妊に係る対面診療への対応可能時間帯 -> 医療機関における緊急避妊にかかる対面診療への対応可能時間帯
    for old_col_index, old_col in column_mapping.items():
        # old_colはoutput_format_list_dfのnameのold_col_indexの値
        new_col = output_format_list_df.loc[old_col_index, 'name']

        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    return df


def reorder_columns(df):
    """
    output_format.csvのname列に記載された順番にdfの列を整理
    """
    desired_order = output_format_list_df['name'].dropna().tolist()
    existing_columns = [col for col in desired_order if col in df.columns]
    missing_columns = [col for col in desired_order if col not in df.columns]

    # 欠落している列を表示
    if missing_columns:
        print(f"Missing columns: {missing_columns}")

    df = df[existing_columns]

    return df


def calculate_check_digit(five_digit_code):
    # 各桁に重みを掛けて合計を求める
    weights = [6, 5, 4, 3, 2]  # 重み
    total = sum(int(digit) * weight for digit,
                weight in zip(five_digit_code, weights))

    # 合計を11で割った余りを求める
    remainder = total % 11

    # 余りからチェックディジットを計算
    if remainder == 0:
        check_digit = 0
    else:
        check_digit = 11 - remainder

    # 特殊ケース
    if check_digit == 10:
        check_digit = 0

    return check_digit


def convert_five_to_six_digit_code(five_digit_code):
    check_digit = calculate_check_digit(five_digit_code)
    new_code = f"{five_digit_code}{check_digit}"
    # print(f"Converted {five_digit_code} to {new_code}")

    return new_code

# usage: python main.py               # output csv files (default)
# usage: python main.py --output-json # output json files
def main(argv):
    if argv[0] == '--output-json':
        print("Exporting to JSON files...")
    else:
        print("Exporting to CSV files...")

    for i, prefecture in enumerate(PREFECTURES, 1):
        prefecture_number_str = str(i).zfill(2)
        prefecture_name = get_prefecture_name(prefecture)
        print(f"PREFECTURE_NUMBER {i}: {prefecture_name} ({prefecture})")

        try:
            opendata_files = os.listdir(f"./data_files/shinryoujo_{i}")
            opendata_file = opendata_files[0]

            file_path = f"./data_files/shinryoujo_{i}/{opendata_file}"
            with pdfplumber.open(file_path) as pdf:
                first_page = pdf.pages[0]
                first_table = first_page.extract_table()
                if first_table is None or len(first_table) < 2:
                    print("No table found.")
                    continue
                headers, data = get_first_page(first_table, prefecture_name)
                df = pd.DataFrame(data, columns=headers)
                clear_change_line(df)

                for page_num in range(1, len(pdf.pages)):
                    page = pdf.pages[page_num]
                    table = page.extract_table()
                    if table:
                        page_df = pd.DataFrame(table, columns=headers)

                        # 「基本情報」「施設名」「医療機関名」を含む行を削除
                        page_df = fix_format_page_df(page_df, 1)
                        clear_change_line(page_df)
                        df = pd.concat([df, page_df], ignore_index=True)

            # 沖縄県と静岡県は『公表の希望の有無』の列を削除
            if prefecture_name in ["沖縄県", "静岡県"]:
                df.drop(df.columns[0], axis=1, inplace=True)

            if "郵便番号" in df.columns:
                # 郵便番号から市区町村を取得
                df["都道府県"], df["市町村"] = zip(
                    *df["郵便番号"].apply(lambda x: postal2location(x) if pd.notna(x) else ("", "")))

                # 郵便番号から市町村コードを取得
                df["市町村コード"] = df["郵便番号"].apply(
                    lambda x: postal2location_code(x) if pd.notna(x) else "")

            if "住所" in df.columns:
                # 住所に都道府県が書いていない行にprefecture_nameを先頭に入れる
                null_prefecture_address = df[df["住所"].str.contains(
                    prefecture_name) == False]
                if not null_prefecture_address.empty:
                    df.loc[null_prefecture_address.index,
                           "住所"] = prefecture_name + null_prefecture_address["住所"]

                # 緯度経度を取得
                df["緯度"], df["経度"] = zip(
                    *df["住所"].apply(lambda x: address_to_coordinates(x) if pd.notna(x) else ("0", "0")))

                # 都道府県、市区町村が空の行に住所から取得した都道府県、市区町村を入れる
                null_prefecture = df[(df["都道府県"] == "") | (df["市町村"] == "")]
                if not null_prefecture.empty:
                    region_locality = null_prefecture["住所"].apply(
                        lambda x: split_japanese_address(x)[:2])
                    region_locality = pd.DataFrame(region_locality.tolist(
                    ), index=null_prefecture.index, columns=["都道府県", "市町村"])
                    df.update(region_locality)
                    # 都道府県、市区町村から市区町村コードを取得
                    df["市町村コード"] = df.apply(
                        lambda x: address2location_code(x["都道府県"], x["市町村"]), axis=1)

            # フォーマットを変換/統一
            df = unify_column_names(df, prefecture_name)

            # 列を並び替える
            df = reorder_columns(df)

            # CSVファイルに出力
            output_file_path = f"./output_files/{prefecture_number_str}_{prefecture}.csv"
            df.to_csv(output_file_path, header=True, index=False)

            if argv[0] == '--output-json':
                # JSONファイルに出力
                prefecture_number_str = str(i).zfill(2)
                output_file_path = f"./output_files/json/{prefecture_number_str}_{prefecture}.json"
                df.to_json(output_file_path, orient='records', force_ascii=False)
            else:
                # CSVファイルに出力
                prefecture_number_str = str(i).zfill(2)
                output_file_path = f"./output_files/{prefecture_number_str}_{prefecture}.csv"
                df.to_csv(output_file_path, header=True, index=False)

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    try:
        if not os.path.exists("./output_files"):
            os.mkdir("./output_files")
        if not os.path.exists("./output_files/json"):
            os.mkdir("./output_files/json")
        main(sys.argv[1:] or [''])
    except Exception as e:
        print(f"Error: {e}")
