import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import butter, filtfilt
import math
from scipy.signal import find_peaks

# Função para aplicar um filtro passa-baixa Butterworth


def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    nyquist_freq = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Função para calcular o ângulo entre três pontos


def calcular_angulo(p1, p2, p3):
    vetor1 = p1 - p2
    vetor2 = p3 - p2
    dot_product = np.dot(vetor1, vetor2)
    mag_v1 = np.linalg.norm(vetor1)
    mag_v2 = np.linalg.norm(vetor2)
    if mag_v1 == 0 or mag_v2 == 0:
        return np.nan
    cos_theta = np.clip(dot_product / (mag_v1 * mag_v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


# Interface do Streamlit
st.title("Cálculo da Angulação do Cotovelo")

# Upload dos arquivos CSV e TXT
uploaded_file = st.file_uploader(
    "Carregue o arquivo CSV contendo os dados dos marcadores", type=["csv"])
uploaded_file2 = st.file_uploader(
    "Carregue o arquivo TXT contendo os dados do smartphone", type=["txt"])
kinem_x = st.number_input('Inserir o valor do translado horizontal da cinemática')
kinem_y = st.number_input('Inserir o valor do translado vertical da cinemática',value=160.0)
smartphone_x = st.number_input('Inserir o valor do translado horizontal do smartphone',value = 5.0)
smartphone_y = st.number_input('Inserir o valor do translado vertical do smartphone',value = 0.0)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if df.shape[1] < 10:
        st.error("O arquivo deve conter pelo menos 10 colunas: tempo, coordenadas do ombro (x, y, z), cotovelo (x, y, z) e punho (x, y, z)")
    else:
        tempo = df.iloc[:, 0]
        ombro = df.iloc[:, 1:4].values
        cotovelo = df.iloc[:, 4:7].values
        punho = df.iloc[:, 7:10].values
        angulos = np.array(
            [calcular_angulo(ombro[i], cotovelo[i], punho[i]) for i in range(len(tempo))])
        resultado = pd.DataFrame(
            {"Tempo": tempo, "Ângulo do Cotovelo (graus)": angulos})

        if uploaded_file2 is not None:
            df_smart = pd.read_csv(uploaded_file2, sep=";")
            tempo_smart = df_smart.iloc[:, 0]
            x, y, z = df_smart.iloc[:, 1].values, df_smart.iloc[:,
                                                                2].values, df_smart.iloc[:, 3].values

            t_ = np.arange(
                start=tempo_smart.iloc[0], stop=tempo_smart.iloc[-1], step=10)
            t_vf = t_ / 1000

            interpf_x = scipy.interpolate.interp1d(
                tempo_smart, x, kind='linear', fill_value="extrapolate")
            interpf_y = scipy.interpolate.interp1d(
                tempo_smart, y, kind='linear', fill_value="extrapolate")
            interpf_z = scipy.interpolate.interp1d(
                tempo_smart, z, kind='linear', fill_value="extrapolate")
            x_vf, y_vf, z_vf = interpf_x(t_), interpf_y(t_), interpf_z(t_)

            def filtrar_sinal(data):
                return butter_lowpass_filter(data, 40, 100, order=4)

            x_vf, y_vf, z_vf = map(filtrar_sinal, [x_vf, y_vf, z_vf])

            accelAngleX = np.arctan(
                y_vf / np.sqrt(x_vf**2 + z_vf**2)) * 180 / math.pi
            angulo = accelAngleX + 90
            
            t_vf = t_vf-smartphone_x
            angulo = angulo+smartphone_y
            peaks, _ = find_peaks(angulo, height=80, distance=100)
            
            # Criar gráfico
            st.write("### Angulação do Cotovelo ao Longo do Tempo")
            fig, ax = plt.subplots()
            ax.plot(tempo, kinem_y - angulos, label="Ângulo do Cotovelo", color='b')
            ax.plot(t_vf, angulo, 'r', label="Ângulo Acelerômetro")
            ax.plot(t_vf[peaks][3:-3], angulo[peaks][3:-3], 'o')
            
            ax.plot([0, 90], [90, 90], '--g')
            ax.plot([0, 90], [87.5, 87.5], '--k')
            ax.plot([0, 90], [85, 85], '--b')

            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Ângulo (graus)")

            st.pyplot(fig)

            csv = resultado.to_csv(index=False).encode('utf-8')
            st.download_button(label="Baixar Dados Processados", data=csv,
                               file_name="angulos_cotovelo.csv", mime="text/csv")
