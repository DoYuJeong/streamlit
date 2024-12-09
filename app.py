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

    try:
        df = pd.read_csv(uploaded_file, usecols=['prop_x', 'prop_y', 'x', 'y', 'sample_id'])
        df['x'] = eval_columns(df['x'])
        df['y'] = eval_columns(df['y'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# 열전 물성이 모두 존재하는 샘플 필터링 함수
def filter_samples_with_all_properties(df, property_mappings):
    property_samples = {}

    # 각 물성별 샘플 ID를 추출
    for prop_key, (properties, _) in property_mappings.items():
        property_samples[prop_key] = df[df['prop_y'].isin(properties)]['sample_id'].unique()

    # 공통 샘플 ID를 계산
    common_samples = set.intersection(*[set(samples) for samples in property_samples.values()])
    return df[df['sample_id'].isin(common_samples)], common_samples

# TEP 그래프 그리기 함수
def plot_TEP(df, sample_id):
    def process_temperature(row):
        if row['prop_x'] == 'Inverse temperature':
            return [1/t if t != 0 else np.nan for t in row['x']]
        return row['x']

    def create_property_df(filtered_df, column_name, transform_func=None):
        if filtered_df.empty:
            return pd.DataFrame(columns=['sample_id', 'temperature', column_name])

        lens = filtered_df['y'].map(len)
        sample_ids = filtered_df['sample_id'].repeat(lens).values
        temperatures = np.concatenate(filtered_df.apply(process_temperature, axis=1).values)
        values = np.concatenate(filtered_df.apply(lambda row: transform_func(row) if transform_func else row['y'], axis=1).values)

        return pd.DataFrame({
            'sample_id': sample_ids,
            'temperature': temperatures,
            column_name: values
        })

    property_mappings = {
        'sigma': (
            ['Electrical conductivity', 'Electrical resistivity'], 
            lambda row: [1/v if v != 0 else np.nan for v in row['y']] if row['prop_y'] == 'Electrical resistivity' else row['y']
        ),
        'alpha': (
            ['Seebeck coefficient', 'thermopower'], 
            None
        ),
        'k': (
            ['Thermal conductivity', 'total thermal conductivity'], 
            None
        ),
        'ZT': (
            ['ZT'], 
            None
        )
    }

    def create_property_dataframes(df, sample_id, property_mappings):
        dataframes = {}
        for column_name, (properties, transform_func) in property_mappings.items():
            filtered_df = df[(df['prop_y'].isin(properties)) & (df['sample_id'] == sample_id)]
            dataframes[column_name] = create_property_df(filtered_df, column_name, transform_func).sort_values(by='temperature')
        return dataframes

    dataframes = create_property_dataframes(df, sample_id, property_mappings)

    df_sigma = dataframes['sigma']
    df_alpha = dataframes['alpha']
    df_k = dataframes['k']
    df_ZT = dataframes['ZT']

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

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = load_and_process_data(uploaded_file)
        if df is not None:
            st.write("Data loaded successfully!")
            property_mappings = {
                'sigma': (
                    ['Electrical conductivity', 'Electrical resistivity'], 
                    None
                ),
                'alpha': (
                    ['Seebeck coefficient', 'thermopower'], 
                    None
                ),
                'k': (
                    ['Thermal conductivity', 'total thermal conductivity'], 
                    None
                ),
                'ZT': (
                    ['ZT'], 
                    None
                )
            }

            filtered_df, common_samples = filter_samples_with_all_properties(df, property_mappings)
            if not common_samples:
                st.error("No samples with all thermoelectric properties found!")
                return

            sample_id = st.selectbox("Select Sample ID (with all properties):", sorted(common_samples))
            if sample_id:
                st.write(f"Filtered DataFrame for Sample ID {sample_id}:")
                st.write(df[df['sample_id'] == sample_id])

                st.write(f"Graphs for Sample ID: {sample_id}")
                plot_TEP(filtered_df, sample_id)
    else:
        st.info("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()


