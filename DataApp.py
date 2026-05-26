import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. Configuração Inicial da Página
st.set_page_config(page_title="Dashboard UNODC - Homicídios", layout="wide")
st.title("📊 Análise e Predição de Homicídios - UNODC")
st.markdown("**Trabalho 01 - Tópicos Especiais em Computação I | Prof. Iális Cavalcante**")

@st.cache_data
def carregar_dados():
    url = f'https://drive.google.com/uc?export=download&id=1LGQWS6zN8yc7coP-HrWm5MygUUyp1afX'
    df = pd.read_excel(url)
    return df

with st.spinner("Carregando base de dados (Dataset)..."):
    df_bruto = carregar_dados()

# 2. Menu Lateral (Sidebar)
st.sidebar.header("Painel de Controle")
menu = st.sidebar.radio(
    "Escolha o módulo a visualizar:",
    ["Análise Exploratória (EDA)", "Modelo de Regressão (Predição)"]
)

# Estilo Padrão para os gráficos Plotly
TEMPLATE_PLOTLY = "plotly_white"

# =====================================================================
# MÓDULO 1: ANÁLISE EXPLORATÓRIA (EDA)
# =====================================================================

if menu == "Análise Exploratória (EDA)":
    st.header("🔍 Análise Exploratória (Estatística Descritiva)")
    
    lista_perguntas = [
        "1. Top 10 países com maiores taxas (2018-2022)",
        "2. Top 10 países com maiores taxas de mulheres (2022)",
        "3. Regiões com mais homicídios (2018-2022)",
        "4. Países com menor taxa em cada sub-região",
        "5. Países com menor número de mortes de mulheres",
        "6. Sub-regiões com as maiores taxas",
        "7. País mais violento em cada continente (2020)",
        "8. País mais violento para mulheres (2021)",
        "9. País com maior volume absoluto acumulado",
        "10. Evolução e média da taxa no Brasil (2013-2022)"
    ]
    
    pergunta = st.selectbox("Selecione a análise:", lista_perguntas)
    st.divider()

    df_taxa = df_bruto[(df_bruto['Unit of measurement'] == 'Rate per 100,000 population') & (df_bruto['VALUE'] < 150)]

    def estilizar_grafico_barra(fig):
        """Aplica estilo padronizado de alta visibilidade aos gráficos de barra"""
        fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.9, textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        fig.update_layout(template=TEMPLATE_PLOTLY, title_font_size=20, xaxis_title_font_size=14, yaxis_title_font_size=14, margin=dict(r=50))
        return fig

    if pergunta == lista_perguntas[0]:
        st.subheader("Quais países apresentam os 10 maiores índices de homicídios nos últimos 5 anos?")
        df_p1 = df_taxa[(df_taxa['Year'] >= 2018) & (df_taxa['Year'] <= 2022) & (df_taxa['Indicator'] == 'Victims of intentional homicide') & (df_taxa['Sex'] == 'Total') & (df_taxa['Age'] == 'Total')]
        top_10 = df_p1.groupby('Country')['VALUE'].mean().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_10, x='VALUE', y='Country', orientation='h', title='Top 10 Taxas Médias (2018-2022)', labels={'VALUE': 'Taxa Média por 100 mil hab.', 'Country': 'País'}, color='VALUE', color_continuous_scale='Reds', text_auto='.1f')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(estilizar_grafico_barra(fig), use_container_width=True)

    elif pergunta == lista_perguntas[1]:
        st.subheader("Quais países apresentam os 10 maiores índices de homicídios de mulheres em 2022?")
        df_p2 = df_taxa[(df_taxa['Year'] == 2022) & (df_taxa['Indicator'] == 'Victims of intentional homicide') & (df_taxa['Sex'] == 'Female') & (df_taxa['Age'] == 'Total')]
        top_10_mulheres = df_p2.groupby('Country')['VALUE'].mean().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_10_mulheres, x='VALUE', y='Country', orientation='h', title='Top 10 Taxas de Homicídios de Mulheres (2022)', labels={'VALUE': 'Taxa por 100 mil mulheres', 'Country': 'País'}, color='VALUE', color_continuous_scale='Purples', text_auto='.1f')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(estilizar_grafico_barra(fig), use_container_width=True)

    elif pergunta == lista_perguntas[2]:
        st.subheader("Quais as regiões com mais homicídios nos últimos 5 anos?")
        df_p3 = df_taxa[(df_taxa['Year'] >= 2018) & (df_taxa['Year'] <= 2022) & (df_taxa['Indicator'] == 'Victims of intentional homicide') & (df_taxa['Sex'] == 'Total') & (df_taxa['Age'] == 'Total')]
        rank_regioes = df_p3.groupby('Region')['VALUE'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(rank_regioes, x='Region', y='VALUE', title='Ranking das Regiões (2018-2022)', labels={'VALUE': 'Taxa Média por 100 mil hab.', 'Region': 'Região'}, color='VALUE', color_continuous_scale='Turbo', text_auto='.1f')
        st.plotly_chart(estilizar_grafico_barra(fig), use_container_width=True)

    elif pergunta == lista_perguntas[3]:
        st.subheader("Quais países com menor número (taxa) de homicídios em cada sub-região?")
        df_geral = df_taxa[(df_taxa['Indicator'] == 'Victims of intentional homicide') & (df_taxa['Sex'] == 'Total') & (df_taxa['Age'] == 'Total')]
        df_subregioes = df_geral.groupby(['Subregion', 'Country'])['VALUE'].mean().reset_index()
        indices_menores = df_subregioes.groupby('Subregion')['VALUE'].idxmin()
        menores_por_subregiao = df_subregioes.loc[indices_menores].sort_values('Subregion').reset_index(drop=True)
        menores_por_subregiao.columns = ['Sub-região', 'País', 'Taxa Média']
        st.dataframe(menores_por_subregiao.style.format({'Taxa Média': '{:.2f}'}).background_gradient(cmap='Greens', subset=['Taxa Média']), use_container_width=True)

    elif pergunta == lista_perguntas[4]:
        st.subheader("Quais países com menor número de morte de mulheres?")
        df_mulheres = df_taxa[(df_taxa['Indicator'] == 'Victims of intentional homicide') & (df_taxa['Sex'] == 'Female') & (df_taxa['Age'] == 'Total') & (df_taxa['VALUE'] > 0)]
        menores_mulheres = df_mulheres.groupby('Country')['VALUE'].mean().sort_values(ascending=True).head(10).reset_index()
        fig = px.bar(menores_mulheres, x='VALUE', y='Country', orientation='h', title='Top 10 Países com Menores Taxas de Homicídios de Mulheres', labels={'VALUE': 'Taxa por 100 mil mulheres', 'Country': 'País'}, color='VALUE', color_continuous_scale='Blues', text_auto='.2f')
        fig.update_layout(yaxis={'categoryorder': 'total descending'})
        st.plotly_chart(estilizar_grafico_barra(fig), use_container_width=True)

    elif pergunta == lista_perguntas[5]:
        st.subheader("Quais as sub-regiões com as maiores taxas de homicídios?")
        df_geral = df_taxa[(df_taxa['Indicator'] == 'Victims of intentional homicide') & (df_taxa['Sex'] == 'Total') & (df_taxa['Age'] == 'Total')]
        ranking_subregioes = df_geral.groupby('Subregion')['VALUE'].mean().sort_values(ascending=False).reset_index().head(10)
        fig = px.bar(ranking_subregioes, x='VALUE', y='Subregion', orientation='h', title='Top 10 Sub-regiões Mais Violentas', labels={'VALUE': 'Taxa Média por 100 mil hab.', 'Subregion': 'Sub-região'}, color='VALUE', color_continuous_scale='YlOrRd', text_auto='.1f')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(estilizar_grafico_barra(fig), use_container_width=True)

    elif pergunta == lista_perguntas[6]:
        st.subheader("Identifique o país com maior número de homicídios em cada continente em 2020.")
        df_2020 = df_taxa[(df_taxa['Year'] == 2020) & (df_taxa['Indicator'] == 'Victims of intentional homicide') & (df_taxa['Sex'] == 'Total') & (df_taxa['Age'] == 'Total')]
        indices_maiores = df_2020.groupby('Region')['VALUE'].idxmax()
        maiores_2020 = df_2020.loc[indices_maiores, ['Region', 'Country', 'VALUE']].sort_values('VALUE', ascending=False).reset_index(drop=True)
        maiores_2020.columns = ['Continente (Região)', 'País', 'Taxa em 2020']
        st.dataframe(maiores_2020.style.format({'Taxa em 2020': '{:.1f}'}).background_gradient(cmap='Reds', subset=['Taxa em 2020']), use_container_width=True)

    elif pergunta == lista_perguntas[7]:
        st.subheader("Qual o país mais violento para as mulheres em 2021?")
        df_2021_mulheres = df_taxa[(df_taxa['Year'] == 2021) & (df_taxa['Indicator'] == 'Victims of intentional homicide') & (df_taxa['Sex'] == 'Female') & (df_taxa['Age'] == 'Total')]
        mais_violento = df_2021_mulheres.sort_values('VALUE', ascending=False).head(1).reset_index(drop=True)
        pais = mais_violento['Country'][0]
        taxa = mais_violento['VALUE'][0]
        st.info(f"🏆 **O país mais violento para mulheres em 2021 foi:**")
        st.metric(label=f"País: {pais}", value=f"{taxa:.1f}", delta="Taxa por 100 mil mulheres", delta_color="off")

    elif pergunta == lista_perguntas[8]:
        st.subheader("Qual o país com maior valor de 'Victims of intentional homicide' (Volume Absoluto)?")
        df_absoluto = df_bruto[(df_bruto['Indicator'] == 'Victims of intentional homicide') & (df_bruto['Unit of measurement'] == 'Counts') & (df_bruto['Sex'] == 'Total') & (df_bruto['Age'] == 'Total')]
        top_volume = df_absoluto.groupby('Country')['VALUE'].sum().sort_values(ascending=False).head(1).reset_index()
        pais = top_volume['Country'][0]
        volume = top_volume['VALUE'][0]
        st.warning(f"🚨 **País com maior acumulado histórico de vítimas no dataset:**")
        st.metric(label=f"País: {pais}", value=f"{int(volume):,} Vítimas", delta="Total Acumulado", delta_color="off")

    elif pergunta == lista_perguntas[9]:
        st.subheader("Qual a média de homicídios no Brasil nos últimos 10 anos (2013-2022)?")
        df_br = df_taxa[(df_taxa['Country'] == 'Brazil') & (df_taxa['Year'] >= 2013) & (df_taxa['Year'] <= 2022) & (df_taxa['Indicator'] == 'Victims of intentional homicide') & (df_taxa['Sex'] == 'Total') & (df_taxa['Age'] == 'Total')].sort_values('Year')
        media_br = df_br['VALUE'].mean()
        st.success(f"🇧🇷 **A taxa média do Brasil neste período é de {media_br:.2f} por 100 mil habitantes.**")
        
        fig2 = px.line(df_br, x='Year', y='VALUE', markers=True, title='Evolução Anual da Taxa no Brasil (2013-2022)', labels={'VALUE': 'Taxa por 100 mil hab.', 'Year': 'Ano'}, text='VALUE')
        fig2.update_traces(textposition="top center", texttemplate='%{text:.1f}', line=dict(color='green', width=3), marker=dict(size=10, line=dict(width=2, color='white')))
        fig2.update_layout(template=TEMPLATE_PLOTLY, xaxis_title_font_size=14, yaxis_title_font_size=14)
        fig2.update_xaxes(dtick=1)
        st.plotly_chart(fig2, use_container_width=True)

# =====================================================================
# MÓDULO 2: MODELO DE REGRESSÃO (PREDIÇÃO)
# =====================================================================

elif menu == "Modelo de Regressão (Predição)":
    st.header("📈 Modelos de Regressão Linear Múltipla e Tendências (2013-2026)")
    st.markdown("Os **10 principais insights preditivos** extraídos do Scikit-Learn. Modelos aprimorados visualmente para destacar projeções.")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌎 1. Global e Brasil", "👥 2. Análise por Sexo", "💰 3. Maiores Economias", "🗺️ 4. Regiões do Mundo", "🎯 5. Preditor Interativo"])
    
    anos_futuros = np.array([2023, 2024, 2025, 2026]).reshape(-1, 1)
    
    df_counts = df_bruto[
        (df_bruto['Indicator'] == 'Victims of intentional homicide') &
        (df_bruto['Unit of measurement'] == 'Counts') &
        (df_bruto['Year'] >= 2013) & (df_bruto['Year'] <= 2022)
    ]

    def plot_trend(df_sub, color_hist, color_pred, title, y_label="Homicídios (Volume)"):
        """Função aprimorada de plotagem de tendência (Regressão)"""
        model = LinearRegression().fit(df_sub[['Year']].values, df_sub['VALUE'].values)
        preds = model.predict(anos_futuros)
        
        fig = go.Figure()
        
        # Dados Históricos (Bolinhas com contorno)
        fig.add_trace(go.Scatter(x=df_sub['Year'], y=df_sub['VALUE'], mode='lines+markers', name='Histórico (Real)', 
                                 line=dict(color=color_hist, width=3), 
                                 marker=dict(size=9, line=dict(width=1.5, color='white'))))
        
        # Previsão (Linha tracejada e marcador de estrela/losango)
        x_pred = np.append(df_sub['Year'].iloc[-1], anos_futuros.flatten()) # Conecta a última linha real com a previsão
        y_pred = np.append(df_sub['VALUE'].iloc[-1], preds)
        
        fig.add_trace(go.Scatter(x=x_pred, y=y_pred, mode='lines+markers', name='Previsão (Regressão)', 
                                 line=dict(dash='dash', color=color_pred, width=3.5),
                                 marker=dict(size=11, symbol='diamond', line=dict(width=1.5, color='white'))))
        
        # Linha Divisória de Previsão
        fig.add_vline(x=2022.5, line_width=2, line_dash="dot", line_color="gray", annotation_text=" Início das Previsões", annotation_position="top right")
        
        fig.update_layout(template=TEMPLATE_PLOTLY, title=title, title_font_size=18, xaxis_title="Ano", yaxis_title=y_label,
                          hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(dtick=1)
        return fig

    # --- ABA 1: GLOBAL E BRASIL ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Tendência Global")
            df_g = df_counts[(df_counts['Sex'] == 'Total') & (df_counts['Age'] == 'Total')].groupby('Year')['VALUE'].sum().reset_index()
            st.plotly_chart(plot_trend(df_g, '#2C3E50', '#E74C3C', "Volume Global de Homicídios"), use_container_width=True)
            
        with col2:
            st.subheader("2. Tendência do Brasil")
            df_br = df_counts[(df_counts['Country'] == 'Brazil') & (df_counts['Sex'] == 'Total') & (df_counts['Age'] == 'Total')].groupby('Year')['VALUE'].sum().reset_index()
            st.plotly_chart(plot_trend(df_br, '#27AE60', '#F39C12', "Volume de Homicídios no Brasil"), use_container_width=True)

    # --- ABA 2: ANÁLISE POR SEXO ---
    with tab2:
        col3, col4 = st.columns(2)
        df_m = df_counts[(df_counts['Sex'] == 'Male') & (df_counts['Age'] == 'Total')].groupby('Year')['VALUE'].sum().reset_index()
        df_f = df_counts[(df_counts['Sex'] == 'Female') & (df_counts['Age'] == 'Total')].groupby('Year')['VALUE'].sum().reset_index()
        
        with col3:
            st.subheader("3. Vítimas Homens")
            st.plotly_chart(plot_trend(df_m, '#2980B9', '#34495E', "Previsão: Homicídios Masculinos"), use_container_width=True)
            
        with col4:
            st.subheader("4. Vítimas Mulheres")
            st.plotly_chart(plot_trend(df_f, '#8E44AD', '#D35400', "Previsão: Homicídios Femininos"), use_container_width=True)

        st.subheader("5. Razão de Vítimas (Feminino / Masculino)")
        df_ratio = pd.merge(df_f, df_m, on='Year', suffixes=('_F', '_M'))
        df_ratio['VALUE'] = df_ratio['VALUE_F'] / df_ratio['VALUE_M']
        st.plotly_chart(plot_trend(df_ratio, '#C0392B', '#7F8C8D', "Razão Histórica e Previsão (Mulheres por Homens)", "Razão F/M"), use_container_width=True)

    # --- ABA 3: TOP 10 ECONOMIAS ---
    with tab3:
        st.subheader("6. Top 10 Economias do Mundo (Volume Absoluto)")
        top_10_countries = ['United States of America', 'China', 'Germany', 'Japan', 'United Kingdom', 'India', 'France', 'Italy', 'Russian Federation', 'Brazil']
        df_eco = df_counts[(df_counts['Country'].isin(top_10_countries)) & (df_counts['Sex'] == 'Total') & (df_counts['Age'] == 'Total')].groupby(['Country', 'Year'])['VALUE'].sum().reset_index()
        
        fig_eco = go.Figure()
        colors = px.colors.qualitative.Prism
        for i, pais in enumerate(top_10_countries):
            df_c = df_eco[df_eco['Country'] == pais]
            if not df_c.empty:
                model_c = LinearRegression().fit(df_c[['Year']].values, df_c['VALUE'].values)
                y_full = np.append(df_c['VALUE'].values, model_c.predict(anos_futuros))
                x_full = np.append(df_c['Year'].values, anos_futuros.flatten())
                fig_eco.add_trace(go.Scatter(x=x_full, y=y_full, mode='lines+markers', name=pais, 
                                             line=dict(color=colors[i % len(colors)], width=2.5),
                                             marker=dict(size=6)))
                
        fig_eco.add_vline(x=2022.5, line_width=2, line_dash="dot", line_color="gray", annotation_text=" Previsões")
        fig_eco.update_layout(template=TEMPLATE_PLOTLY, title="Comparativo das 10 Maiores Economias (2013-2026)", hovermode="x unified")
        st.plotly_chart(fig_eco, use_container_width=True)

    # --- ABA 4: REGIÕES ---
    with tab4:
        st.subheader("7. Tendência Geral das Maiores Regiões")
        df_reg = df_counts[(df_counts['Sex'] == 'Total') & (df_counts['Age'] == 'Total')].groupby(['Region', 'Year'])['VALUE'].sum().reset_index()
        top_regions = df_reg.groupby('Region')['VALUE'].sum().nlargest(7).index.tolist()
        
        fig_reg = go.Figure()
        colors_reg = px.colors.qualitative.Vivid
        for i, regiao in enumerate(top_regions):
            df_r = df_reg[df_reg['Region'] == regiao]
            if not df_r.empty:
                model_r = LinearRegression().fit(df_r[['Year']].values, df_r['VALUE'].values)
                y_full_r = np.append(df_r['VALUE'].values, model_r.predict(anos_futuros))
                x_full_r = np.append(df_r['Year'].values, anos_futuros.flatten())
                fig_reg.add_trace(go.Scatter(x=x_full_r, y=y_full_r, mode='lines+markers', name=regiao,
                                             line=dict(color=colors_reg[i % len(colors_reg)], width=2.5), marker=dict(size=6)))
        
        fig_reg.add_vline(x=2022.5, line_width=2, line_dash="dot", line_color="gray", annotation_text=" Previsões")
        fig_reg.update_layout(template=TEMPLATE_PLOTLY, hovermode="x unified")
        st.plotly_chart(fig_reg, use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            st.subheader("8. Homicídios Femininos (Por Região)")
            df_reg_f = df_counts[(df_counts['Sex'] == 'Female') & (df_counts['Age'] == 'Total')].groupby(['Region', 'Year'])['VALUE'].sum().reset_index()
            top_regions_f = df_reg_f.groupby('Region')['VALUE'].sum().nlargest(5).index.tolist()
            
            fig_reg_f = go.Figure()
            for i, reg in enumerate(top_regions_f):
                df_rf = df_reg_f[df_reg_f['Region'] == reg]
                model_rf = LinearRegression().fit(df_rf[['Year']].values, df_rf['VALUE'].values)
                y_full_rf = np.append(df_rf['VALUE'].values, model_rf.predict(anos_futuros))
                fig_reg_f.add_trace(go.Scatter(x=np.append(df_rf['Year'].values, anos_futuros.flatten()), y=y_full_rf, mode='lines', name=reg, line=dict(width=3)))
            fig_reg_f.add_vline(x=2022.5, line_width=1, line_dash="dot", line_color="gray")
            fig_reg_f.update_layout(template=TEMPLATE_PLOTLY, hovermode="x unified")
            st.plotly_chart(fig_reg_f, use_container_width=True)
            
        with col6:
            st.subheader("9. Dispersão Regional (Histórico)")
            fig_box = px.box(df_reg, x='Region', y='VALUE', color='Region', points="all")
            fig_box.update_layout(template=TEMPLATE_PLOTLY, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

    # --- ABA 5: PREDITOR INTERATIVO ---
    with tab5:
        st.subheader("10. Simulador Individual de Taxas (Linear vs Polinomial)")
        col7, col8 = st.columns(2)
        with col7:
            paises_disponiveis = df_bruto['Country'].dropna().unique()
            pais_alvo = st.selectbox("Escolha o País:", sorted(paises_disponiveis), index=sorted(paises_disponiveis).index('Brazil'))
        with col8:
            ano_limite = st.slider("Prever até:", 2023, 2030, 2026)

        df_modelo = df_bruto[
            (df_bruto['Country'] == pais_alvo) &
            (df_bruto['Indicator'] == 'Victims of intentional homicide') &
            (df_bruto['Unit of measurement'] == 'Rate per 100,000 population') &
            (df_bruto['Sex'] == 'Total') & (df_bruto['Age'] == 'Total')
        ].sort_values('Year')
        
        if len(df_modelo) > 3:
            X = df_modelo['Year'].values.reshape(-1, 1)
            y = df_modelo['VALUE'].values
            
            modelo_linear = LinearRegression().fit(X, y)
            poly = PolynomialFeatures(degree=2)
            modelo_poly = LinearRegression().fit(poly.fit_transform(X), y)
            
            anos_futuros_int = np.arange(2023, ano_limite + 1).reshape(-1, 1)
            pred_linear = modelo_linear.predict(anos_futuros_int)
            pred_poly = modelo_poly.predict(poly.transform(anos_futuros_int))
            
            fig_indiv = go.Figure()
            # Linha Real
            fig_indiv.add_trace(go.Scatter(x=df_modelo['Year'], y=y, mode='lines+markers', name='Histórico Real', line=dict(color='blue', width=3), marker=dict(size=9)))
            
            # Conectores e Previsões
            x_pred_int = np.append(df_modelo['Year'].iloc[-1], anos_futuros_int.flatten())
            y_pred_lin = np.append(y[-1], pred_linear)
            y_pred_poly = np.append(y[-1], pred_poly)
            
            fig_indiv.add_trace(go.Scatter(x=x_pred_int, y=y_pred_lin, mode='lines+markers', name='Regressão Linear', line=dict(dash='dash', color='red', width=3), marker=dict(symbol='square', size=9)))
            fig_indiv.add_trace(go.Scatter(x=x_pred_int, y=y_pred_poly, mode='lines+markers', name='Regressão Polinomial (Grau 2)', line=dict(dash='dot', color='green', width=3), marker=dict(symbol='diamond', size=9)))
            
            fig_indiv.add_vline(x=2022.5, line_width=2, line_dash="dot", line_color="gray", annotation_text=" Início das Previsões")
            fig_indiv.update_layout(template=TEMPLATE_PLOTLY, title=f"Previsões de Taxa (100k hab.) - {pais_alvo}", hovermode="x unified", xaxis_title="Ano", yaxis_title="Taxa por 100k hab.")
            fig_indiv.update_xaxes(dtick=1)
            st.plotly_chart(fig_indiv, use_container_width=True)
            
            df_tabela = pd.DataFrame({'Ano': anos_futuros_int.flatten(), 'Predição Linear': pred_linear.round(2), 'Predição Polinomial': pred_poly.round(2)})
            st.dataframe(df_tabela.style.background_gradient(cmap='Greys'), use_container_width=True)
        else:
            st.warning("Não há dados históricos suficientes na base (mínimo de 3 anos) para rodar o modelo preditivo para este país.")