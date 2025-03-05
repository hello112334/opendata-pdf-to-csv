import pandas as pd
import os
import jageocoder
import requests
import logger
import xml.etree.ElementTree as ET



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

##### main.pyから持ってきた関数群ここまで #####


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
##### main_pyから持ってきて、ちょっと回収した関数群ここまで #######

jageocoder.init(url='https://jageocoder.info-proto.com/jsonrpc')


def main(i):
    opendata_files = os.listdir(f"./data_files/yakkyoku_{i}")
    opendata_file = opendata_files[0]

    file_path = f"./data_files/yakkyoku_{i}/{opendata_file}"
    
    df = pd.read_excel(file_path,skiprows=[0,1,2]) # タイトル行を削除する。都道府県ごとに変えるかも

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

    #施設系
    df = df.rename(columns={"薬局名":"施設_名称"})
    df = df.rename(columns={"開局時間":"施設_開局時間"})

    # 連絡先系
    df = df.rename(columns={"電話番号":"連絡先_電話番号"}) 
    df = df.rename(columns={"FAX番号":"連絡先_FAX番号"}) 
    df = df.rename(columns={"郵便番号":"連絡先_郵便番号"}) 
    df = df.rename(columns={"薬局所在地（市郡区以降）":"連絡先_住所"}) 


    #施設_市区町村コード
    #住所_都道府県
    #住所_市区町村（郡）
    df["住所_都道府県"], df["住所_市区町村（郡）"], df["施設_市区町村コード"] = zip(
            *df["連絡先_郵便番号"].apply(lambda x: postal2location(x) if pd.notna(x) else ("", "", "")))
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
    "施設_開局時間",
    "施設_市区町村コード",
    "住所_都道府県",
    "住所_市区町村（郡）",
    "住所_緯度",
    "住所_経度",
    "連絡先_郵便番号",
    "連絡先_住所",
    "連絡先_電話番号",
    "連絡先_FAX番号",
    "時間外_対応の有無",
    "時間外_電話番号",    
    ]]

    #エラー

    output_file_path = f"./output_files/yakkyoku_{i}.csv"
    df.to_csv(output_file_path)


main(29)
