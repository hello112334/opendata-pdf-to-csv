import pandas as pd
import pdfplumber
import os
import requests
import re
import xml.etree.ElementTree as ET
import sys
import codecs

import jageocoder
from enum import Enum
from datetime import datetime

from urllib.parse import urlparse

# 緯度経度ダブルチェック
import geopandas as gpd
from shapely.geometry import Point
# 半角変換
import unicodedata
# ログ出力
import logging
from logging import getLogger, basicConfig, INFO

# ==================================================================
# Function
# ==================================================================


def load_prefectures(csv_file):
    """
    都道府県番号と名前の対応辞書をCSVファイルから読み込む
    """
    df = pd.read_csv(csv_file, dtype={
                     'number': int, 'name': str, 'english_name': str})
    return {row['english_name']: row['name'] for _, row in df.iterrows()}, list(df['english_name'])


def get_prefecture_name(prefecture_english_name):
    """
    都道府県の英語名から日本語名を取得
    """
    return prefectures.get(prefecture_english_name, "")


def address_to_coordinates(address):
    """
    住所から緯度経度を取得
    note: (token)エラーが発生した場合はjageocoder_searchに切り替える
    """
    if not address:
        return 0, 0
    try:
        base_url = "http://geocode.csis.u-tokyo.ac.jp/cgi-bin/simple_geocode.cgi?charset=UTF8&addr="
        url = base_url + requests.utils.quote(str(address))
        latitude, longitude = 0, 0
        response = requests.get(url)
        if response.status_code == 200:
            xml_content = response.text
            xml_content = xml_content.replace("\n", "")
            root = ET.fromstring(xml_content)

            # 小数点以下第6位まで取得
            longitude = round(
                float(root.findtext(".//candidate/longitude")), 6)
            latitude = round(float(root.findtext(".//candidate/latitude")), 6)

    except Exception as e:
        logger.error(f"{address_to_coordinates} {e}")
        latitude, longitude = jageocoder_search(address)

    return latitude, longitude


def jageocoder_search(address):
    """
    住所から緯度経度を取得 (jageocoderを使用)
    address_to_coordinatesがエラーの場合に使用
    """
    if not address:
        return 0, 0

    address = str(address)
    result = jageocoder.search(address)

    if result['candidates']:
        # 最初の候補から緯度経度を取得
        latitude = result['candidates'][0]['y']
        longitude = result['candidates'][0]['x']

        # 緯度経度の範囲を確認する
        if (-90 <= latitude <= 90) and (-180 <= longitude <= 180):
            return round(latitude, 6), round(longitude, 6)

    return 0, 0


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
    郵便番号から市区町村名と市区町村コードを取得
    """
    if pd.isna(postal_code):
        return "", "", ""

    postal_code = postal_code.replace("-", "")

    # 1. ken_all.csvと突き合わせ
    prefecture, city = "", ""
    city_code = ""
    for _, row in address_df.iterrows():
        if row["postal"].strip() == postal_code:
            prefecture = row["prefecture"].strip()
            city = row["city"].strip()
            city_code = row["jis"].strip()
            break

    # 2. 個別事業所データと突き合わせ
    if prefecture == "" and city == "":
        for _, row in jigyosyo_df.iterrows():
            if row["postal"].strip() == postal_code:
                prefecture = row["prefecture"].strip()
                city = row["city"].strip()
                city_code = row["jis"].strip()
                break

    # 5桁を6桁に変換
    if city_code != "":
        tmp_code = f'{city_code:05}'  # 0埋めで5文字
        city_code = convert_five_to_six_digit_code(tmp_code)

    return prefecture, city, city_code


def is_valid_url(url):
    if not urlparse(url).scheme:
        url = 'https://' + url
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def is_accessible_url(url):
    if not urlparse(url).scheme:
        url = 'https://' + url
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        # ステータスコードが200番台または300番台であればアクセス可能と判断
        return response.status_code < 400
    except requests.RequestException:
        return False


def delete_title(df):
    """
    大分県に不要なタイトルがあるため削除
    """
    if df.iloc[0, 0] == "緊急避妊に係る診療が可能な産婦人科医療機関等一覧":
        return df.drop(df.index[:1])
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
    return delete_headers(delete_title(df), line_number)


def zenkaku_to_hankaku_regex(text):
    """
    全角を半角に変換する関数
    """
    if text:
        text = unicodedata.normalize('NFKC', text)
    return text


def clear_change_line(df):
    """
    行の表記を統一する処理
    """
    # 改行コードを削除
    df[['施設_名称', '連絡先_電話番号', '連絡先_FormURL', '連絡先_郵便番号', '連絡先_住所', '緊急避妊に係る診療が可能な産婦人科医療機関等_産科、婦人科又は産婦人科の標榜の有無', '緊急避妊に係る診療が可能な産婦人科医療機関等_常時の緊急避妊薬の在庫の有無']].replace(r'\r\n|\r|\n', '', regex=True, inplace=True)

    # "を削除
    df.replace('"', '', regex=True, inplace=True)

    # 時間表記の「~」を「-」に変換
    hyphens = ['-', '˗', 'ᅳ', '᭸', '‐', '‑', '‒', '–', '—', '―', '⁃', '⁻', '−', '▬', '─', '━', '➖', 'ー', 'ㅡ', '﹘', '﹣', '－', 'ｰ', '𐄐', '𐆑', ' ']
    df.replace('~', '-', regex=True, inplace=True)
    df.replace('〜', '-', regex=True, inplace=True)
    df.replace(hyphens, '-', regex=True, inplace=True)

    # 全角を半角に変換する
    df = df.apply(lambda x: x.map(zenkaku_to_hankaku_regex))

    # データが2つ未満の行は不要な可能性が高いので行を削除 & 列名に欠損値がある場合も列ごと削除
    df.dropna(axis=0, thresh=2, inplace=True)

    # 郵便番号の欄に「〒」がある場合は削除
    df["連絡先_郵便番号"] = df["連絡先_郵便番号"].str.replace("〒", "")

    # 何もない行を削除(ex:静岡県)
    # 名称、住所、郵便番号がない行で判定する
    df.dropna(subset=["施設_名称", "連絡先_住所", "連絡先_郵便番号"], how="all", inplace=True)

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


def unify_column_names(df, prefecture_name):
    """
    フォーマットのカラムを変換/統一
    """
    if prefecture_name not in output_format_list_df.columns:
        logger.error(
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
        logger.warning(f"Missing columns: {missing_columns}")

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

    return new_code


def check_location_in_japan(row):
    """
    指定された座標が指定された都道府県と市区町村に属するかを確認する関数。

    Parameters:
    row (pandas.Series): データフレームの1行

    Returns:
    str: エラーメッセージ（エラーがない場合は空文字列）
    """
    latitude = row["住所_緯度"]
    longitude = row["住所_経度"]
    expected_prefecture = row["住所_都道府県"]

    # 緯度経度が0の場合はすでにエラーが設定されているためスキップ
    if latitude == 0 or longitude == 0 or row["住所_都道府県"] == "":
        return row["エラー"]

    # 指定された座標のポイントを作成
    point = Point(longitude, latitude)

    # 座標が含まれる都道府県を検索
    matching_area = gdf[gdf.contains(point)]

    if matching_area.empty:
        return add_error_message(row, ERROR_LIST.LAT_LON_MISMATCH.value)

    # 抽出された都道府県
    extracted_prefecture = matching_area.iloc[0]['N03_001']  # 都道府県名

    # 提供された都道府県と比較
    if extracted_prefecture != expected_prefecture:
        return add_error_message(row, ERROR_LIST.LAT_LON_MISMATCH.value)

    return row["エラー"]


def add_error_message(row, error_message):
    """
    エラーメッセージを追加する
    """
    if "エラー" not in row.keys():
        row["エラー"] = ""

    row["エラー"] = f"{row['エラー']}, {error_message}" if row["エラー"] else error_message
    logger.error(
        f"[{error_message}] {row['施設_名称']}: 都道府県:{row['住所_都道府県']}, 市区町村:{row['住所_市区町村（郡）']} 緯度経度({row['住所_緯度']},{row['住所_経度']}) 連絡先_FormURL:{row['連絡先_FormURL']}")

    return row["エラー"]


def main(i, prefecture, argv):
    """
    usage: python main.py               # output csv files (default)
    usage: python main.py --output-json # output json files
    """

    prefecture_number_str = str(i).zfill(2)
    prefecture_name = get_prefecture_name(prefecture)
    logger.info(f"PREFECTURE_NUMBER {i}: {prefecture_name} ({prefecture})")

    try:
        opendata_files = os.listdir(f"./data_files/shinryoujo_{i}")
        opendata_file = opendata_files[0]

        file_path = f"./data_files/shinryoujo_{i}/{opendata_file}"
        with pdfplumber.open(file_path) as pdf:
            first_page = pdf.pages[0]
            first_table = first_page.extract_table()
            if first_table is None or len(first_table) < 2:
                logger.warning("No table found.")
                return
            headers, data = get_first_page(first_table, prefecture_name)
            df = pd.DataFrame(data, columns=headers)

            for page_num in range(1, len(pdf.pages)):
                page = pdf.pages[page_num]
                table = page.extract_table()
                if table:
                    page_df = pd.DataFrame(table, columns=headers)

                    # 「基本情報」「施設名」「医療機関名」を含む行を削除
                    page_df = fix_format_page_df(page_df, 1)
                    df = pd.concat([df, page_df], ignore_index=True)

        # フォーマットを変換/統一
        df = unify_column_names(df, prefecture_name)

        # データの整理
        df = clear_change_line(df)

        # 追加列の初期化
        df["エラー"] = ""
        df["住所_都道府県"] = ""
        df["住所_市区町村（郡）"] = ""
        df["施設_市区町村コード"] = ""
        df["住所_緯度"], df["住所_経度"] = 0, 0

        # 沖縄県と静岡県は『公表の希望の有無』の列を削除
        if prefecture_name in ["沖縄県", "静岡県"]:
            df.drop(df.columns[0], axis=1, inplace=True)

        if "連絡先_郵便番号" in df.columns:
            # 郵便番号から市区町村を取得
            df["住所_都道府県"], df["住所_市区町村（郡）"], df["施設_市区町村コード"] = zip(
                *df["連絡先_郵便番号"].apply(lambda x: postal2location(x) if pd.notna(x) else ("", "", "")))

            # エラー列を更新し、郵便番号エラーを追加する
            df["エラー"] = df.apply(lambda x: add_error_message(
                x, ERROR_LIST.POST_CODE.value) if x["住所_都道府県"] == "" and x["住所_市区町村（郡）"] == "" else x["エラー"], axis=1)

        if "連絡先_住所" in df.columns:
            # 住所に都道府県が書いていない行にprefecture_nameを先頭に入れる
            null_prefecture_address = df[df["連絡先_住所"].str.contains(
                prefecture_name) == False]
            if not null_prefecture_address.empty:
                df.loc[null_prefecture_address.index,
                       "連絡先_住所"] = prefecture_name + null_prefecture_address["連絡先_住所"]

            # 緯度経度を取得
            # address_to_coordinates
            df["住所_緯度"], df["住所_経度"] = zip(
                *df["連絡先_住所"].apply(lambda x: address_to_coordinates(x) if pd.notna(x) else ("0", "0")))

            # エラー
            df["エラー"] = df.apply(
                lambda x: add_error_message(x, ERROR_LIST.LAT_LON_ERROR.value) if x["住所_緯度"] == "0" and x["住所_経度"] == "0" else x["エラー"], axis=1)

        if "連絡先_FormURL" in df.columns:
            # URLが有効かどうかを確認
            df["エラー"] = df.apply(
                lambda x: add_error_message(x, ERROR_LIST.URL_INVALID.value)
                if x["連絡先_FormURL"] != '' and not is_valid_url(x["連絡先_FormURL"])
                else x["エラー"],
                axis=1
            )

            df["エラー"] = df.apply(
                lambda x: add_error_message(x, ERROR_LIST.URL_EXPIRED.value)
                if x["連絡先_FormURL"] != '' and not is_accessible_url(x["連絡先_FormURL"])
                else x["エラー"],
                axis=1
            )

        # 列を並び替える
        df = reorder_columns(df)

        # GeoJSONで緯度経度をチェックする
        df["エラー"] = df.apply(check_location_in_japan, axis=1)

        if argv[0] == '--output-json':
            # JSONファイルに出力
            prefecture_number_str = str(i).zfill(2)
            output_file_path = f"./output_files/json/{prefecture_number_str}_{prefecture}.json"
            df.to_json(output_file_path, orient='records',
                       force_ascii=False)
        else:
            # CSVファイルに出力
            prefecture_number_str = str(i).zfill(2)
            output_file_path = f"./output_files/{prefecture_number_str}_{prefecture}.csv"
            df.to_csv(output_file_path, header=True, index=False)

    except Exception as e:
        logger.error(e)


# ==================================================================
# Parameters
# ==================================================================

# エラーリスト
class ERROR_LIST(Enum):
    POST_CODE = "郵便番号"
    # 個別事業所データと医療機関データに郵便番号がない

    LAT_LON_ERROR = "緯度経度(検索エラー)"
    # 緯度経度でgeojsonから都道府県を抽出した際に、空欄になったのか郵便番号から都道府県名と比較して不一致

    LAT_LON_MISMATCH = "緯度経度(不一致)"
    # センターから取得した緯度経度とGoogle APIの緯度経度と比較し、合わない

    GOOGLE_LAT_LON_ERROR = "緯度経度(Googleエラー)"
    # Google APIで緯度経度を取得する際、返却値がない場合や2値以上の返却値がある

    URL_INVALID = "URL(無効なURL)"
    # 産科、婦人科又は産婦人科の標榜の有無のカラム内にpやmが入り込んでいる(?)

    URL_EXPIRED = "URL(リンク切れ)"
    # URLにPingを打ち、404などのエラーが返った


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    try:
        # フォルダ初期化
        if not os.path.exists("./output_files"):
            os.mkdir("./output_files")
        if not os.path.exists("./output_files/json"):
            os.mkdir("./output_files/json")
        if not os.path.exists("./logs"):
            os.mkdir("./logs")

        # ログ設定
        # yyyymmdd format
        current_date = datetime.now().strftime("%Y%m%d")
        basicConfig(filename=f"logs/{current_date}.log", filemode='a',
                    format='[%(asctime)s]%(levelname)-7s: %(message)s',
                    level=INFO)

        # コンソール（標準出力）に出力するハンドラを作成
        logger = logging.getLogger(__name__)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        # start
        logger.info(f"{'='*10} START {'='*10}")

        # load config
        logger.info("Initializing jageocoder...")

        # geojson
        geojson_file_path = "./data_files/JPGIS2014/N03-20240101_prefecture.geojson"
        gdf = gpd.read_file(geojson_file_path)  # GeoJSONデータを読み込む

        # jageocoder
        jageocoder.init(url='https://jageocoder.info-proto.com/jsonrpc')

        # 都道府県の辞書と英語名のリストを読み込む
        prefectures, PREFECTURES = load_prefectures('./table/prefectures.csv')

        # csv出力フォマード+転換リストを読み込む
        output_format_list_df = pd.read_csv(
            './table/output_format.csv', header=0, dtype=str)

        # CSVファイルを読み込み、郵便番号をキー、市区町村名を値とする辞書を作成
        address_df = pd.read_csv(
            './data_files/ken_all/utf_ken_all.csv', header=None, dtype=str,
            names=["jis", "old", "postal", "prefecture_kana", "city_kana", "town_kana", "prefecture", "city", "town", "multi", "koaza", "block", "double", "update", "reason"])

        # 事業所CSVファイルを読み込む
        # 文字コードにShift-JISでないものも混じっているようで、エラーは無視する
        with codecs.open('./data_files/jigyosyo/JIGYOSYO.CSV', "r", "Shift-JIS", "ignore") as file:
            # カラム名はken_allと合わせる
            jigyosyo_df = pd.read_csv(
                file, header=None, dtype=str, encoding="shift-jis",
                names=["jis", "jigyosyo_kana", "jigyosyo", "prefecture", "city", "town", "detail", "postal", "old", "branch", "type", "multi", "diff"])

        # 出力確認
        argv = sys.argv[1:] or ['']
        if argv[0] == '--output-json':
            logger.info("Exporting to JSON files...")
        else:
            logger.info("Exporting to CSV files...")

        # 通常処理
        # テストモード --test は北海道だけ実行
        for index, prefecture in enumerate(PREFECTURES[0:], 1):
            if "--test" in argv:
                if index == 1:
                    main(index, prefecture, argv)
                break
            else:
                main(index, prefecture, argv)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        logger.info(f"{'='*10}  END  {'='*10}")
