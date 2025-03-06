import pandas as pd
import os
import jageocoder
import requests
import xml.etree.ElementTree as ET
from enum import Enum
import logging
from logging import getLogger, basicConfig, INFO
from shapely.geometry import Point
import geopandas as gpd


# コンソール（標準出力）に出力するハンドラを作成
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# geojson
geojson_file_path = "./data_files/JPGIS2014/N03-20240101_prefecture.geojson"
gdf = gpd.read_file(geojson_file_path)  # GeoJSONデータを読み込む


##### main.pyから持ってきた関数群ここから #####
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
#        logger.error(f"{address_to_coordinates} {e}")
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



import codecs
# 事業所CSVファイルを読み込む
# 文字コードにShift-JISでないものも混じっているようで、エラーは無視する
with codecs.open('./data_files/jigyosyo/JIGYOSYO.CSV', "r", "Shift-JIS", "ignore") as file:
    # カラム名はken_allと合わせる
    jigyosyo_df = pd.read_csv(
        file, header=None, dtype=str, encoding="shift-jis",
        names=["jis", "jigyosyo_kana", "jigyosyo", "prefecture", "city", "town", "detail", "postal", "old", "branch", "type", "multi", "diff"])

# CSVファイルを読み込み、郵便番号をキー、市区町村名を値とする辞書を作成
address_df = pd.read_csv(
    './data_files/ken_all/utf_ken_all.csv', header=None, dtype=str,
    names=["jis", "old", "postal", "prefecture_kana", "city_kana", "town_kana", "prefecture", "city", "town", "multi", "koaza", "block", "double", "update", "reason"])


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

import unicodedata


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

##### main.pyから持ってきた関数群ここまで #####


# 都道府県の辞書と英語名のリストを読み込む
prefectures, PREFECTURES = load_prefectures('./table/prefectures.csv')


##### main_pyから持ってきて、ちょっと回収した関数群ここから #######
def zenkaku_to_hankaku_regex(text):
    """
    全角を半角に変換する関数
    """
    if text:
        text = unicodedata.normalize('NFKC', str(text))
    return text


# 表記統一処理
def clear_change_line(df):
    """
    行の表記を統一する処理
    """
    # 改行コードを削除
    df[['施設_名称', '連絡先_電話番号',  '連絡先_郵便番号', '連絡先_住所']].replace(r'\r\n|\r|\n', '', regex=True, inplace=True)

    # "を削除
    df.replace('"', '', regex=True, inplace=True)
    
    # エクセル改行コードを削除
    df.replace('_x000D_', ' ', regex=True, inplace=True)

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
        f"[{error_message}] {row['施設_名称']}: 都道府県:{row['住所_都道府県']}, 市区町村:{row['住所_市区町村（郡）']} 緯度経度({row['住所_緯度']},{row['住所_経度']})")

    return row["エラー"]


##### main_pyから持ってきて、ちょっと回収した関数群ここまで #######
jageocoder.init(url='https://jageocoder.info-proto.com/jsonrpc')


def main(index, prefecture):
    try:
        opendata_files = os.listdir(f"./data_files/yakkyoku_{index}")
        opendata_file = opendata_files[0]

        file_path = f"./data_files/yakkyoku_{index}/{opendata_file}"

        if index != 23 or index != 29:
            df = pd.read_excel(file_path, skiprows=[0, 1, 2])

        # カラム名の改行を削除
        for _col in df.columns:
            df = df.rename(columns={_col : _col.replace("\n","")})

        # 全般の改行を削除
        for _col in df.columns:
            if df[_col].dtype == object:
                df[_col] = df[_col].map(lambda x : str(x).replace("\n","")) 

        # 連番カラム削除
        df = df.drop("連番",axis=1)
        # 都道府県番号カラム削除
        df = df.drop("都道府県番号",axis=1)
        df = df.drop("都道府県",axis=1)
        # 薬剤師カラム削除
        df = df.drop("研修を修了した薬剤師氏名",axis=1)
        df = df.drop("研修を修了した薬剤師数",axis=1)

        df.columns = [
            "施設_名称",
            "連絡先_郵便番号",
            "連絡先_住所",
            "連絡先_電話番号",
            "連絡先_FAX番号",
            "施設_開局時間",
            "時間外対応の有無",
            "時間外の電話番号"
        ]

        # エラー列の追加
        df["エラー"] = ""
        df["住所_緯度"], df["住所_経度"] = 0, 0

        #施設_市区町村コード
        #住所_都道府県
        #住所_市区町村（郡）
        df["住所_都道府県"], df["住所_市区町村（郡）"], df["施設_市区町村コード"] = zip(
                *df["連絡先_郵便番号"].apply(lambda x: postal2location(x) if pd.notna(x) else ("", "", "")))

        # エラー列を更新し、郵便番号エラーを追加する
        df["エラー"] = df.apply(lambda x: add_error_message(
            x, ERROR_LIST.POST_CODE.value) if x["住所_都道府県"] == "" and x["住所_市区町村（郡）"] == "" else x["エラー"], axis=1)
        #住所_緯度
        #住所_経度
        ######## ToDo確認・住所の文字列、丸投げして緯度経度出しているが問題ないか？ #########
        ######## ToDo確認・緯度経度の正当性、確認するか？できるか？ #########
        df["住所_緯度"], df["住所_経度"] = zip(
            *df["連絡先_住所"].apply(lambda x: address_to_coordinates(x) if pd.notna(x) else ("0", "0")))

        # 時間外系
        df = df.rename(columns={"時間外の電話番号":"時間外_電話番号"}) 
        df.loc[df["時間外_電話番号"] == "nan" , "時間外_電話番号"] = "" # 「nan」という文字列が入っていたので、空文字に変換
        df = df.rename(columns={"時間外対応の有無":"時間外_対応の有無"}) 

        # 表記整え
        df = clear_change_line(df)
        # 必要なカラムだけ、必要な順番で取得
        df = df[[
            "施設_名称",
            "施設_市区町村コード",
            "住所_都道府県",
            "住所_市区町村（郡）",
            "住所_緯度",
            "住所_経度",
            "連絡先_電話番号",
            "連絡先_FAX番号",
            "連絡先_郵便番号",
            "連絡先_住所",
            "施設_開局時間",
            "時間外_対応の有無",
            "時間外_電話番号",
            "エラー",
        ]]

        print(df.head())

        # エラー
        df["エラー"] = df.apply(
            lambda x: add_error_message(x, ERROR_LIST.LAT_LON_ERROR.value) if x["住所_緯度"] == "0" and x["住所_経度"] == "0" else x["エラー"], axis=1)

        # GeoJSONで緯度経度をチェックする
        df["エラー"] = df.apply(check_location_in_japan, axis=1)

        # CSVファイルに出力
        prefecture_number_str = str(index).zfill(2)
        prefecture_name = get_prefecture_name(prefecture)
        logger.info(f"PREFECTURE_NUMBER {index}: {prefecture_name} ({prefecture})")
        output_file_path = f"./output_files/薬局_{prefecture_number_str}_{prefecture}.csv"
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


#for index, prefecture in enumerate(PREFECTURES[0:], 1):
#    main(index, prefecture)

main(47, PREFECTURES[46])
