import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. Configuração Inicial da Página
st.set_page_config(page_title="Dashboard UNODC - Homicídios", layout="wide")
st.title("📊 Análise e Predição de Homicídios - UNODC")
st.markdown("**Trabalho 01 - Tópicos Especiais em Computação I | Prof. Iális Cavalcante**")


@st.cache_data
def carregar_dados():
    url = f'https://drive.google.com/uc?export=download&id=1t_F3EwUGrK0D-TuuW73XXDyE6v_WlDPA'
    df = pd.read_excel(url)
    return df

with st.spinner("Carregando base de dados (Dataset)"):
    df_bruto = carregar_dados()

# 3. Menu Lateral (Sidebar)
st.sidebar.header("Painel de Controlo")
menu = st.sidebar.radio(
    "Escolha o módulo a visualizar:",
    ["Análise Exploratória (EDA)", "Modelo de Regressão (Predição)"]
)


#  ANÁLISE EXPLORATÓRIA (EDA)

if menu == "Análise Exploratória (EDA)":
    st.header("🔍 Análise Exploratória (Estatística Descritiva)")
    
    pergunta = st.selectbox(
        "Selecione a análise:",
        ["Top 10 Países Mais Violentos (2018-2022)", "Evolução da Taxa no Brasil (2013-2022)"]
    )
    
    if pergunta == "Top 10 Países Mais Violentos (2018-2022)":
        st.subheader("Quais os países com as maiores taxas médias de homicídios nos últimos 5 anos?")
        
        # Filtro como fizemos no Colab
        df_taxa = df_bruto[
            (df_bruto['Indicator'] == 'Victims of intentional homicide') &
            (df_bruto['Unit of measurement'] == 'Rate per 100,000 population') &
            (df_bruto['Sex'] == 'Total') & (df_bruto['Age'] == 'Total') &
            (df_bruto['Year'] >= 2018)
        ]
        
        top10 = df_taxa.groupby('Country')['VALUE'].mean().sort_values(ascending=False).head(10).reset_index()
        
        # Gráfico dinâmico com Plotly
        fig = px.bar(
            top10, x='VALUE', y='Country', orientation='h',
            title='Top 10 Taxas Médias (2018-2022)',
            labels={'VALUE': 'Taxa por 100 mil hab.', 'Country': 'País'},
            color='VALUE', color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    elif pergunta == "Evolução da Taxa no Brasil (2013-2022)":
        st.subheader("Qual a média de homicídios no Brasil nos últimos 10 anos?")
        
        df_br = df_bruto[
            (df_bruto['Country'] == 'Brazil') &
            (df_bruto['Indicator'] == 'Victims of intentional homicide') &
            (df_bruto['Unit of measurement'] == 'Rate per 100,000 population') &
            (df_bruto['Sex'] == 'Total') & (df_bruto['Age'] == 'Total')
        ].sort_values('Year')
        
        media_br = df_br['VALUE'].mean()
        st.info(f"**A taxa média do Brasil neste período é de {media_br:.2f} por 100 mil habitantes.**")
        
        fig2 = px.line(
            df_br, x='Year', y='VALUE', markers=True,
            title='Evolução Anual no Brasil',
            labels={'VALUE': 'Taxa por 100 mil hab.', 'Year': 'Ano'}
        )
        st.plotly_chart(fig2, use_container_width=True)


#  MODELO DE REGRESSÃO

elif menu == "Modelo de Regressão (Predição)":
    st.header("📈 Modelo de Regressão e Predição (2023-2026)")
    st.markdown("O modelo utiliza **Scikit-Learn** para aprender com o histórico e prever tendências futuras.")
    
    # Filtros para o modelo
    col1, col2 = st.columns(2)
    with col1:
        paises_disponiveis = df_bruto['Country'].dropna().unique()
        pais_alvo = st.selectbox("Escolha o País para treinar o modelo:", sorted(paises_disponiveis), index=sorted(paises_disponiveis).index('Brazil'))
    with col2:
        ano_limite = st.slider("Prever até qual ano?", 2023, 2030, 2026)

    # Filtrar dados do país escolhido
    df_modelo = df_bruto[
        (df_bruto['Country'] == pais_alvo) &
        (df_bruto['Indicator'] == 'Victims of intentional homicide') &
        (df_bruto['Unit of measurement'] == 'Rate per 100,000 population') &
        (df_bruto['Sex'] == 'Total') & (df_bruto['Age'] == 'Total')
    ].sort_values('Year')
    
    if len(df_modelo) > 3:
        # Preparar dados para o Scikit-Learn
        X = df_modelo['Year'].values.reshape(-1, 1)
        y = df_modelo['VALUE'].values
        
        # Treinar Regressão Linear
        modelo_linear = LinearRegression()
        modelo_linear.fit(X, y)
        
        # Treinar Regressão Polinomial (Grau 2)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        modelo_poly = LinearRegression()
        modelo_poly.fit(X_poly, y)
        
        # Criar base de anos futuros para prever
        anos_futuros = np.arange(2023, ano_limite + 1).reshape(-1, 1)
        anos_futuros_poly = poly.transform(anos_futuros)
        
        pred_linear = modelo_linear.predict(anos_futuros)
        pred_poly = modelo_poly.predict(anos_futuros_poly)
        
        # Juntar histórico com previsões para plotar
        df_historico = pd.DataFrame({'Ano': df_modelo['Year'], 'Taxa': y, 'Tipo': 'Histórico (Real)'})
        df_futuro_lin = pd.DataFrame({'Ano': anos_futuros.flatten(), 'Taxa': pred_linear, 'Tipo': 'Previsão Linear'})
        df_futuro_poly = pd.DataFrame({'Ano': anos_futuros.flatten(), 'Taxa': pred_poly, 'Tipo': 'Previsão Polinomial'})
        
        df_grafico = pd.concat([df_historico, df_futuro_lin, df_futuro_poly])
        
        # Gráfico dinâmico
        st.subheader(f"Linha de Tendência e Previsão: {pais_alvo}")
        fig_reg = px.line(
            df_grafico, x='Ano', y='Taxa', color='Tipo', markers=True,
            title=f"Previsões de Taxa de Homicídios - {pais_alvo}",
            labels={'Taxa': 'Taxa por 100 mil hab.'}
        )
        st.plotly_chart(fig_reg, use_container_width=True)
        
        # Mostrar tabela de resultados futuros
        st.write("### Valores Previstos")
        df_tabela = pd.DataFrame({
            'Ano': anos_futuros.flatten(),
            'Previsão Linear': pred_linear.round(2),
            'Previsão Polinomial (Curva)': pred_poly.round(2)
        })
        st.dataframe(df_tabela, use_container_width=True)
        
    else:
        st.warning("Não há dados históricos suficientes para este país na base para treinar o modelo.")