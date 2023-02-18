import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import PIL
import pytesseract
import os
os.environ['TESSDATA_PREFIX'] = 'Extract Tabular Data/tessdata_dir/'


def optimizeDf(old_df: pd.DataFrame) -> pd.DataFrame:
    df = old_df[["left", "top", "width", "text"]]
    df['left+width'] = df['left'] + df['width']
    df = df.sort_values(by=['top'], ascending=True)
    df = df.groupby(['top', 'left+width'], sort=False)['text'].sum().unstack('left+width')
    df = df.reindex(sorted(df.columns), axis=1).dropna(how='all').dropna(axis='columns', how='all')
    df = df.fillna('')
    return df

def mergeDfColumns(old_df: pd.DataFrame, threshold: int = 10, rotations: int = 5) -> pd.DataFrame: # threshold was 10
    df = old_df.copy()
    for j in range(0, rotations):
        new_columns = {}
        old_columns = df.columns
        i = 0
        while i < len(old_columns):
            if i < len(old_columns) - 1:
                if any(old_columns[i+1] == old_columns[i] + x for x in range(1, threshold)):
                    new_col = df[old_columns[i]].astype(str) + df[old_columns[i+1]].astype(str)
                    new_columns[old_columns[i+1]] = new_col
                    i = i + 1
                else:
                    new_columns[old_columns[i]] = df[old_columns[i]]
            else:
                new_columns[old_columns[i]] = df[old_columns[i]]
            i += 1
            df = pd.DataFrame.from_dict(new_columns).replace('', np.nan).dropna(axis='columns', how='all').replace(np.nan, '')
    return df

def mergeDfRows(old_df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    new_df = old_df.iloc[:1]
    for i in range(1, len(old_df)):
        if abs(old_df.index[i] - old_df.index[i - 1]) < threshold:
            new_df.iloc[-1] = new_df.iloc[-1].astype(str) + old_df.iloc[i].astype(str)
        else:
            new_df = new_df.append(old_df.iloc[i])
    return new_df.reset_index(drop=True)

def cleanDf(df):
    # Remove columns with all cells holding the same value and its length is 0 or 1
    df = df.loc[:, (df != df.iloc[0]).any()]
    # Remove rows with empty cells or cells with only the '|' symbol
    df = df[(df != '|') & (df != '=') & (df != '') & (pd.notnull(df))]
    # Remove columns with only empty cells
    df = df.dropna(axis=1, how='all')
    return df.fillna('')

print(pytesseract.get_tesseract_version())
print(pytesseract.get_languages())

"""
Best Rsults: --psm 12 --oem 1
History:
 8) --psm 12 --oem 1 --dpi 3000         -> eng 80%
 7) --psm 12 --oem 2                    -> eng 90%, heb 78%-10%+5% -> ',' and '.' it cant decide between the two
 6) --psm 12 --oem 1                    -> eng 95%, heb 78%-10%+5% -> ',' and '.' it cant decide between the two
 5) --psm 12 --oem 0                    -> eng 85%
 4) --psm 12                            -> eng 90%, heb 75%-15+3%%
 3) --psm 6                             -> 40%
 2) --psm 5                             -> 10% 
 1) --psm 11                            -> eng 85%, heb 70%+-15%
URL: https://muthu.co/all-tesseract-ocr-options/
"""
special_config = '--psm 12 --oem 1'
languages_ = "eng"

image_path = "Model Implementation/DummyDatabase/predictions/image_crop.png"

img_pl=PIL.Image.open(image_path)

data = pytesseract.image_to_data(img_pl, lang=languages_, output_type='data.frame', config=special_config)

data_imp_sort = optimizeDf(data.copy())

df_new_col = mergeDfColumns(data_imp_sort.copy())

merged_row_df = mergeDfRows(df_new_col.copy(), threshold = 5)

cleaned_df = cleanDf(merged_row_df.copy())
