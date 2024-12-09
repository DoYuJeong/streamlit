import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# 데이터 로드 및 처리 함수
def load_and_process_data(uploaded_file, sample_id=None):
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

        # 특정 샘플 ID만 필터링
        if sample_id is not None:
            df = df[df['sample_id'] == sample_id]

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# 그래프 그리기 함수
def plot_TEP(df, sample_id):
    # 온도 변환 함수
    def process_temperature(row):
        if row['prop_x'] == 'Inverse temperature':
            return [1/t if t != 0 else np.nan for t in row['x']]
        return row['x']

    # 데이터프레임 생성 함수
    def create_property_df(prop_y_list, column_name, transform_func=None):
        # prop_y 조건에 맞는 데이터 필터링
        new_df = df[(df['prop_x'].isin(['Temperature', 'Inverse temperature'])) &
                    (df['prop_y'].isin(prop_y_list)) &
                    (df['sample_id'] == sample_id)].copy()

        if new_df.empty:
            return pd.DataFrame(columns=['sample_id', 'temperature', column_name])  # 빈 데이터프레임 반환

        # 리스트 길이 계산
        lens = new_df['y'].map(len)
        sample_ids = new_df['sample_id'].repeat(lens).values

        # 온도 데이터 처리
        temperatures = np.concatenate(new_df.apply(process_temperature, axis=1).values)

        # 물성 데이터 변환 (필요시 transform_func 적용)
        values = np.concatenate(new_df.apply(lambda row: transform_func(row) if transform_func else row['y'], axis=1).values)

        # 최종 데이터프레임 생성
        return pd.DataFrame({
            'sample_id': sample_ids,
            'temperature': temperatures,
            column_name: values
        })

    # 변환 함수 정의
    def transform_sigma(row):
        if row['prop_y'] == 'Electrical resistivity':
            # Electrical resistivity의 역수를 취해 Electrical conductivity로 변환
            return [1/v if v != 0 else np.nan for v in row['y']]
        return row['y']  # Electrical conductivity는 그대로 사용

    # 데이터프레임 생성
    df_sigma = create_property_df(['Electrical conductivity', 'Electrical resistivity'], 'sigma', transform_sigma).sort_values(by='temperature')
    df_alpha = create_property_df(['Seebeck coefficient', 'thermopower'], 'alpha').sort_values(by='temperature')
    df_k = create_property_df(['Thermal conductivity', 'total thermal conductivity'], 'k').sort_values(by='temperature')
    df_ZT = create_property_df(['ZT'], 'ZT').sort_values(by='temperature')

    # 그래프 그리기
    figsize = (10, 8)
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    if not df_sigma.empty:
        ax1.plot(df_sigma['temperature'], df_sigma['sigma'], marker='o', linestyle='-', color='m')
        ax1.set_title('Sigma')
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel(r'$\sigma$ $[S/cm]$')
        ax1.grid(True)

    if not df_alpha.empty:
        ax2.plot(df_alpha['temperature'], df_alpha['alpha'] * 1e6, marker='o', linestyle='-', color='g')
        ax2.set_title('Alpha')
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel(r'$\alpha$ $[\mu V/K]$')
        ax2.grid(True)

    if not df_k.empty:
        ax3.plot(df_k['temperature'], df_k['k'], marker='o', linestyle='-', color='r')
        ax3.set_title('K')
        ax3.set_xlabel('Temperature')
        ax3.set_ylabel(r'$k$ $[W/(m·K)]$')
        ax3.grid(True)

    if not df_ZT.empty:
        ax4.plot(df_ZT['temperature'], df_ZT['ZT'], marker='o', linestyle='-', color='b')
        ax4.set_title(r'$ZT$')
        ax4.set_xlabel('Temperature')
        ax4.set_ylabel(r'$ZT$')
        ax4.grid(True)

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
            # 열전 물성이 모두 존재하는 샘플 필터링
            filtered_df, common_samples = filter_samples_with_all_properties(df)
            if not common_samples:
                st.error("No samples with all thermoelectric properties found!")
                return

            # 샘플 ID 선택
            sample_id = st.selectbox("Select Sample ID (with all properties):", sorted(common_samples))
            if sample_id:
                # 샘플 ID에 해당하는 데이터 추출
                df_data = df[df['sample_id'] == sample_id]
                st.write(f"Filtered DataFrame for sample_id {sample_id}:")
                st.write(df_data)  # Streamlit에서 데이터프레임 출력

                # 그래프 그리기
                st.write(f"Graphs for Sample ID: {sample_id}")
                plot_graphs(sample_id, filtered_df)
    else:
        st.info("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()


