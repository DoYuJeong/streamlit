# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

link = st.link_button("Go to 20241101_starrydata2 download", url="https://figshare.com/projects/Starrydata_datasets/155129")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# 데이터 로드 및 처리 함수
def load_and_process_data(uploaded_file):
    def eval_columns(col):
        try:
            return col.apply(ast.literal_eval)
        except Exception as e:
            st.error(f"Error parsing column values: {e}")
            return col

    # 업로드된 파일에서 데이터 로드
    try:
        df = pd.read_csv(uploaded_file, usecols=['prop_x', 'prop_y', 'x', 'y', 'sample_id'])
        df['x'] = eval_columns(df['x'])
        df['y'] = eval_columns(df['y'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# 열전 물성이 모두 존재하는 샘플 필터링 함수
def filter_samples_with_all_properties(df):
    # 각 열전 물성에 해당하는 데이터 필터링
    seebeck_samples = df[df['prop_y'] == 'Seebeck coefficient']['sample_id'].unique()
    conductivity_samples = df[df['prop_y'] == 'Electrical conductivity']['sample_id'].unique()
    thermal_samples = df[df['prop_y'] == 'Thermal conductivity']['sample_id'].unique()
    ZT_samples = df[df['prop_y'] == 'ZT']['sample_id'].unique()

    # 모든 열전 물성이 존재하는 샘플 ID 교집합
    common_samples = set(seebeck_samples) & set(conductivity_samples) & set(thermal_samples) & set(ZT_samples)
    
    return df[df['sample_id'].isin(common_samples)], common_samples

# 데이터프레임 생성 함수
def create_property_df(df, sample_id, prop_y, column_name):
    new_df = df[(df['prop_x'] == 'Temperature') & (df['prop_y'] == prop_y) & (df['sample_id'] == sample_id)].copy()
    if new_df.empty:
        return pd.DataFrame(columns=['sample_id', 'temperature', column_name])  # 빈 데이터프레임 반환
    lens = new_df['y'].map(len)
    sample_ids = new_df['sample_id'].repeat(lens).values
    temperatures = np.concatenate(new_df['x'].values)
    values = np.concatenate(new_df['y'].values)
    return pd.DataFrame({
        'sample_id': sample_ids,
        'temperature': temperatures,
        column_name: values
    })

# 그래프 그리기 함수
def plot_graphs(sample_id, df):
    # 데이터 생성
    df_sigma = create_property_df(df, sample_id, 'Electrical conductivity', 'sigma')
    df_alpha = create_property_df(df, sample_id, 'Seebeck coefficient', 'alpha')
    df_k = create_property_df(df, sample_id, 'Thermal conductivity', 'k')
    df_ZT = create_property_df(df, sample_id, 'ZT', 'ZT')

    # 그래프 설정
    figsize = (10, 8)
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Conductivity
    if not df_sigma.empty:
        axs[0, 0].plot(df_sigma['temperature'], df_sigma['sigma'], marker='o', linestyle='-', color='m')
        axs[0, 0].set_title(r'$\sigma$ vs Temperature')
        axs[0, 0].set_xlabel('Temperature')
        axs[0, 0].set_ylabel(r'$\sigma$ $[S/cm]$')
        axs[0, 0].grid(True)

    # Alpha
    if not df_alpha.empty:
        axs[0, 1].plot(df_alpha['temperature'], df_alpha['alpha'] * 1e6, marker='o', linestyle='-', color='g')
        axs[0, 1].set_title(r'$\alpha$ vs Temperature')
        axs[0, 1].set_xlabel('Temperature')
        axs[0, 1].set_ylabel(r'$\alpha$ $[\mu V/K]$')
        axs[0, 1].grid(True)

    # Thermal Conductivity
    if not df_k.empty:
        axs[1, 0].plot(df_k['temperature'], df_k['k'], marker='o', linestyle='-', color='r')
        axs[1, 0].set_title(r'$k$ vs Temperature')
        axs[1, 0].set_xlabel('Temperature')
        axs[1, 0].set_ylabel(r'$k$ $[W/(m·K)]$')
        axs[1, 0].grid(True)

    # ZT
    if not df_ZT.empty:
        axs[1, 1].plot(df_ZT['temperature'], df_ZT['ZT'], marker='o', linestyle='-', color='b')
        axs[1, 1].set_title(r'$ZT$ vs Temperature')
        axs[1, 1].set_xlabel('Temperature')
        axs[1, 1].set_ylabel(r'$ZT$')
        axs[1, 1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)  # Streamlit에서 그래프 표시

# Streamlit 앱
def main():
    st.title("Thermoelectric Property Dashboard")
    st.write("Upload your CSV file to get started.")

    # 파일 업로드
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = load_and_process_data(uploaded_file)  # 파일 업로드 후 처리
        if df is not None:
            st.write("Data loaded successfully!")
            st.write(df.head())

            # 열전 물성이 모두 존재하는 샘플 필터링
            filtered_df, common_samples = filter_samples_with_all_properties(df)
            if not common_samples:
                st.error("No samples with all thermoelectric properties found!")
                return

            # 샘플 ID 선택
            sample_id = st.selectbox("Select Sample ID (with all properties):", sorted(common_samples))
            if sample_id:
                st.write(f"Graphs for Sample ID: {sample_id}")
                plot_graphs(sample_id, filtered_df)
    else:
        st.info("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()



