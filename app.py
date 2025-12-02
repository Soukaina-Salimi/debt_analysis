# app.py - Code complet corrigé
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ==================== CLASSES DE MODÈLES ====================

class DebtPredictor:
    def __init__(self):
        self.methods = {
            'moyenne_mobile': self.moving_average_forecast,
            'regression_lineaire': self.linear_regression_forecast,
            'croissance_moyenne': self.average_growth_forecast,
            'exponentiel_lisse': self.exponential_smoothing_forecast
        }
    
    def prepare_time_series(self, country_data, target_var='DT_DOD_ALLC_CD'):
        """Prépare une série temporelle propre"""
        if country_data.empty:
            return None
        
        # Trier par année
        ts = country_data.sort_values('year')[['year', target_var]].copy()
        
        # Nettoyer les valeurs aberrantes
        ts_clean = self.remove_outliers(ts, target_var)
        
        # S'assurer que les valeurs sont positives
        min_val = ts_clean[target_var].min()
        if min_val <= 0:
            offset = abs(min_val) + 1
            ts_clean[f'{target_var}_adj'] = ts_clean[target_var] + offset
            target = f'{target_var}_adj'
        else:
            ts_clean[f'{target_var}_adj'] = ts_clean[target_var]
            target = f'{target_var}_adj'
        
        return ts_clean, target
    
    def remove_outliers(self, df, column):
        """Supprime les outliers avec la méthode IQR"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return df
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    
    def moving_average_forecast(self, ts, target, years_ahead=5, window=3):
        """Prédiction par moyenne mobile"""
        if len(ts) < window:
            return None
        
        # Calculer la moyenne mobile
        ma_values = ts[target].rolling(window=window).mean()
        
        # Dernière valeur de la moyenne mobile
        last_ma = ma_values.iloc[-1]
        
        # Tendance basée sur les dernières valeurs
        recent_values = ts[target].tail(window).values
        if len(recent_values) >= 2:
            trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
        else:
            trend = 0
        
        # Générer les prédictions
        predictions = []
        for i in range(1, years_ahead + 1):
            pred = last_ma + trend * i
            predictions.append(max(pred, ts[target].min() * 0.5))
        
        return predictions
    
    def linear_regression_forecast(self, ts, target, years_ahead=5):
        """Prédiction par régression linéaire simple"""
        if len(ts) < 3:
            return None
        
        X = ts['year'].values.reshape(-1, 1)
        y = ts[target].values
        
        # Régression linéaire
        slope, intercept = np.polyfit(ts['year'], ts[target], 1)
        
        # Prédictions
        last_year = ts['year'].max()
        predictions = []
        
        for i in range(1, years_ahead + 1):
            pred = slope * (last_year + i) + intercept
            predictions.append(max(pred, ts[target].min() * 0.5))
        
        return predictions
    
    def average_growth_forecast(self, ts, target, years_ahead=5):
        """Prédiction basée sur la croissance moyenne"""
        if len(ts) < 2:
            return None
        
        # Calculer les taux de croissance annuels
        ts = ts.sort_values('year')
        values = ts[target].values
        
        growth_rates = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                growth = (values[i] - values[i-1]) / abs(values[i-1])
                growth_rates.append(growth)
        
        if not growth_rates:
            return None
        
        # Moyenne des taux de croissance (limité à [-0.1, 0.2])
        avg_growth = np.mean(growth_rates)
        avg_growth = max(min(avg_growth, 0.2), -0.1)  # Entre -10% et +20%
        
        # Dernière valeur
        last_value = values[-1]
        
        # Prédictions
        predictions = []
        for i in range(1, years_ahead + 1):
            pred = last_value * ((1 + avg_growth) ** i)
            predictions.append(pred)
        
        return predictions
    
    def exponential_smoothing_forecast(self, ts, target, years_ahead=5, alpha=0.3):
        """Lissage exponentiel simple"""
        if len(ts) < 3:
            return None
        
        values = ts[target].values
        
        # Lissage exponentiel
        smoothed = [values[0]]
        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[i-1])
        
        # Tendance basée sur les valeurs lissées
        if len(smoothed) >= 2:
            trend = (smoothed[-1] - smoothed[-2])
        else:
            trend = 0
        
        # Prédictions
        predictions = []
        last_smoothed = smoothed[-1]
        
        for i in range(1, years_ahead + 1):
            pred = last_smoothed + trend * i
            predictions.append(max(pred, ts[target].min() * 0.5))
        
        return predictions
    
    def ensemble_forecast(self, ts, target, years_ahead=5):
        """Combinaison de plusieurs méthodes (ensemble)"""
        predictions_all = {}
        
        for method_name, method_func in self.methods.items():
            pred = method_func(ts, target, years_ahead)
            if pred is not None and len(pred) == years_ahead:
                predictions_all[method_name] = pred
        
        if not predictions_all:
            return None
        
        # Moyenne des prédictions
        pred_matrix = np.array(list(predictions_all.values()))
        ensemble_pred = pred_matrix.mean(axis=0)
        
        return ensemble_pred, predictions_all

class DebtRiskClassifier:
    def __init__(self):
        self.risk_thresholds = {
            'Faible': 30,    # < 30% du PIB (approximation)
            'Modéré': 60,    # 30-60% du PIB
            'Élevé': 90,     # 60-90% du PIB
            'Critique': 100  # > 90% du PIB
        }
    
    def assess_risk(self, debt_ratio, debt_growth):
        """Évalue le niveau de risque"""
        if debt_ratio < self.risk_thresholds['Faible']:
            return 'Faible'
        elif debt_ratio < self.risk_thresholds['Modéré']:
            return 'Modéré' if debt_growth < 10 else 'Élevé'
        elif debt_ratio < self.risk_thresholds['Élevé']:
            return 'Élevé' if debt_growth < 15 else 'Critique'
        else:
            return 'Critique'
    
    def calculate_debt_indicators(self, df):
        """Calcule les indicateurs de risque de dette"""
        indicators = {}
        
        for country in df['country_name'].unique():
            country_data = df[df['country_name'] == country].sort_values('year')
            
            if len(country_data) < 2:
                continue
            
            # Ratio Dette approximatif (dette par habitant * 1000 / revenu estimé)
            if 'DT_DOD_ALLC_CD' in country_data.columns:
                avg_debt = country_data['DT_DOD_ALLC_CD'].mean()
                avg_pop = country_data['SP_POP_TOTL'].mean() if 'SP_POP_TOTL' in country_data.columns else 1
                
                # Approximation du ratio dette/PIB
                debt_per_capita = avg_debt / avg_pop if avg_pop > 0 else 0
                debt_gdp_ratio = min(debt_per_capita / 1000 * 100, 200)  # Normalisation
                
                # Taux de croissance de la dette
                if len(country_data) > 1:
                    debt_growth = country_data['DT_DOD_ALLC_CD'].iloc[-1] / country_data['DT_DOD_ALLC_CD'].iloc[0]
                    debt_growth = ((debt_growth ** (1/len(country_data))) - 1) * 100
                else:
                    debt_growth = 0
                
                indicators[country] = {
                    'debt_gdp_ratio': debt_gdp_ratio,
                    'debt_growth': debt_growth,
                    'risk_level': self.assess_risk(debt_gdp_ratio, debt_growth)
                }
        
        return pd.DataFrame(indicators).T

# ==================== FONCTIONS D'ANALYSE ====================

def plot_debt_evolution(df, selected_countries):
    """Graphique d'évolution de la dette"""
    fig = go.Figure()
    
    for country in selected_countries:
        country_data = df[df['country_name'] == country].sort_values('year')
        if len(country_data) > 0:
            fig.add_trace(go.Scatter(
                x=country_data['year'],
                y=country_data['DT_DOD_ALLC_CD'],
                mode='lines+markers',
                name=country,
                hovertemplate=f"{country}<br>Année: %{{x}}<br>Dette: %{{y:,.0f}}<extra></extra>"
            ))
    
    fig.update_layout(
        title='Évolution de la Dette par Pays',
        xaxis_title='Année',
        yaxis_title='Dette Totale (USD)',
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

def analyze_correlations(df, country):
    """Analyse des corrélations pour un pays"""
    country_data = df[df['country_name'] == country]
    
    # Variables disponibles pour l'analyse de corrélation
    available_vars = []
    potential_vars = [
        'DT_DOD_ALLC_CD', 'SP_POP_GROW', 'SP_URB_TOTL_IN_ZS',
        'co2_emissions', 'access_to_electricity',
        'SE_TER_ENRR', 'SP_DYN_LE00_IN', 'SP_POP_TOTL'
    ]
    
    for var in potential_vars:
        if var in country_data.columns:
            available_vars.append(var)
    
    if len(available_vars) < 2:
        return None, None
    
    # Calcul de la matrice de corrélation
    numeric_data = country_data[available_vars].apply(pd.to_numeric, errors='coerce')
    corr_matrix = numeric_data.corr()
    
    # Création du heatmap
    fig = px.imshow(
        corr_matrix,
        title=f'Matrice de Corrélation - {country}',
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1,
        labels=dict(color="Corrélation")
    )
    
    # Ajouter les valeurs sur le heatmap
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            fig.add_annotation(
                x=j, y=i,
                text=f"{corr_matrix.iloc[i, j]:.2f}",
                showarrow=False,
                font=dict(color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white")
            )
    
    return fig, corr_matrix

# Remplacer la fonction generate_predictions par cette version améliorée :
def generate_robust_predictions(df, country, target_var='DT_DOD_ALLC_CD'):
    """Version sécurisée des prédictions qui évite toutes les erreurs"""
    try:
        # Vérifications de base
        if country not in df['country_name'].unique():
            return {'error': f"Pays '{country}' non trouvé dans les données"}
        
        country_data = df[df['country_name'] == country].copy()
        
        if len(country_data) < 2:
            return {'error': f"Seulement {len(country_data)} année(s) de données pour {country}"}
        
        # Vérifier que la variable cible existe
        if target_var not in country_data.columns:
            return {'error': f"Variable '{target_var}' non trouvée"}
        
        # Trier par année
        country_data = country_data.sort_values('year')
        
        # Nettoyer les données
        debt_values = pd.to_numeric(country_data[target_var], errors='coerce')
        country_data[target_var] = debt_values.fillna(method='ffill').fillna(method='bfill')
        
        if country_data[target_var].isnull().all():
            return {'error': f"Aucune donnée valide pour {target_var}"}
        
        # Vérifier les valeurs négatives
        if (country_data[target_var] < 0).any():
            # Ajouter un offset pour rendre toutes les valeurs positives
            min_value = country_data[target_var].min()
            if min_value <= 0:
                offset = abs(min_value) + 1
                country_data[f'{target_var}_adj'] = country_data[target_var] + offset
                target = f'{target_var}_adj'
            else:
                target = target_var
        else:
            target = target_var
        
        # Méthode simple : croissance moyenne
        years = country_data['year'].values
        values = country_data[target].values
        
        if len(values) < 2:
            return {'error': "Données insuffisantes pour calculer la croissance"}
        
        # Calculer la croissance historique
        growth_rates = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                growth = (values[i] - values[i-1]) / abs(values[i-1])
                growth_rates.append(growth)
        
        if not growth_rates:
            avg_growth = 0.03  # 3% par défaut
        else:
            avg_growth = np.mean(growth_rates)
            # Limiter la croissance à des valeurs raisonnables
            avg_growth = max(min(avg_growth, 0.15), -0.05)
        
        # Prédictions pour 5 ans
        future_years = list(range(years[-1] + 1, years[-1] + 6))
        last_value = values[-1]
        
        predictions = []
        for i in range(1, 6):
            pred = last_value * ((1 + avg_growth) ** i)
            predictions.append(float(pred))
        
        # Retransformer si nécessaire
        if '_adj' in target:
            predictions = [p - offset for p in predictions]
            last_actual_value = country_data[target_var].iloc[-1]
        else:
            last_actual_value = last_value
        
        # Déterminer la tendance
        if len(predictions) >= 2:
            if predictions[-1] > predictions[0]:
                trend = 'croissante'
            elif predictions[-1] < predictions[0]:
                trend = 'décroissante'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'success': True,
            'future_years': future_years,
            'predictions': predictions,
            'last_actual_year': int(years[-1]),
            'last_actual_value': float(last_actual_value),
            'estimated_growth_pct': avg_growth * 100,
            'trend': trend,
            'data_points': len(country_data),
            'method': 'Croissance moyenne simple',
            'historical_years': years.tolist(),
            'historical_values': country_data[target_var].tolist()
        }
        
    except Exception as e:
        return {
            'error': f"Erreur lors de la génération des prédictions: {str(e)}",
            'success': False
        }
# ==================== APPLICATION STREAMLIT ====================

def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Analyse de la Dette Africaine",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #3B82F6;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #3B82F6;
            margin-bottom: 1rem;
        }
        .risk-low { color: #10B981; }
        .risk-medium { color: #F59E0B; }
        .risk-high { color: #EF4444; }
        .risk-critical { color: #7C3AED; }
        </style>
    """, unsafe_allow_html=True)
    
    # En-tête
    st.markdown('<h1 class="main-header">📊 Analyse de la Dette en Afrique</h1>', unsafe_allow_html=True)
    st.markdown("Dashboard interactif d'analyse et de prédiction de la dette des pays africains")
    
    # Chargement des données
    try:
        df = pd.read_csv('global_dataset.csv')
        st.sidebar.success(f"✅ Données chargées: {len(df)} lignes, {len(df['country_name'].unique())} pays")
    except:
        st.error("❌ Erreur de chargement des données. Vérifiez que le fichier 'global_dataset.csv' est dans le même dossier.")
        st.stop()
    
    # Sidebar - Paramètres
    with st.sidebar:
        st.header("⚙️ Paramètres d'analyse")
        
        # Filtre des pays
        all_countries = sorted(df['country_name'].unique())
        selected_countries = st.multiselect(
            "Sélectionner les pays",
            all_countries,
            default=['Algeria', 'Egypt', 'Cameroon'][:min(3, len(all_countries))]
        )
        
        # Filtre des années
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        selected_years = st.slider(
            "Période d'analyse",
            min_year, max_year,
            (min_year, max_year)
        )
        
        # Variable cible
        debt_variables = [col for col in df.columns if 'DOD' in col or 'debt' in col.lower()]
        selected_debt_var = st.selectbox(
            "Variable de dette à analyser",
            debt_variables,
            index=debt_variables.index('DT_DOD_ALLC_CD') if 'DT_DOD_ALLC_CD' in debt_variables else 0
        )
        
        st.divider()
        
        # Informations sur les données
        with st.expander("📊 Informations sur les données"):
            st.write(f"**Période:** {min_year} - {max_year}")
            st.write(f"**Pays disponibles:** {len(all_countries)}")
            st.write(f"**Variables:** {len(df.columns)}")
            
            # Aperçu des données
            if st.button("Aperçu des données"):
                st.dataframe(df.head(), use_container_width=True)
    
    # Filtrage des données
    filtered_df = df[
        (df['year'] >= selected_years[0]) & 
        (df['year'] <= selected_years[1])
    ]
    
    if selected_countries:
        filtered_df = filtered_df[filtered_df['country_name'].isin(selected_countries)]
    
    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Tableau de Bord", 
        "📈 Analyse par Pays",
        "🔮 Prédictions", 
        "📊 Comparaisons",
        "📋 Recommandations"
    ])
    
    # ==================== ONGLET 1: TABLEAU DE BORD ====================
    with tab1:
        st.markdown('<h2 class="sub-header">📊 Vue d\'Ensemble</h2>', unsafe_allow_html=True)
        
        # KPI en ligne
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_debt = filtered_df[selected_debt_var].mean()
            st.metric(
                "Dette Moyenne",
                f"${avg_debt:,.0f}",
                help="Valeur moyenne de la dette sur la période sélectionnée"
            )
        
        with col2:
            if len(filtered_df) > 1:
                grouped = filtered_df.groupby('year')[selected_debt_var].mean()
                if len(grouped) > 1:
                    growth = ((grouped.iloc[-1] / grouped.iloc[0]) ** (1/(len(grouped)-1)) - 1) * 100
                else:
                    growth = 0
            else:
                growth = 0
            st.metric(
                "Croissance Annuelle",
                f"{growth:.1f}%",
                delta=f"{growth:.1f}%",
                delta_color="inverse" if growth > 10 else "normal"
            )
        
        with col3:
            total_countries = len(filtered_df['country_name'].unique())
            st.metric(
                "Pays Analysés",
                total_countries,
                help="Nombre de pays dans la sélection"
            )
        
        with col4:
            total_years = selected_years[1] - selected_years[0] + 1
            st.metric(
                "Période",
                f"{total_years} ans",
                help="Durée de la période d'analyse"
            )
        
        # Graphique d'évolution
        st.markdown("#### Évolution de la Dette")
        if selected_countries:
            fig = plot_debt_evolution(filtered_df, selected_countries)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Veuillez sélectionner au moins un pays")
        
        # Top 5 des pays par dette
        st.markdown("#### Classement par Dette")
        if len(filtered_df) > 0:
            recent_year = filtered_df['year'].max()
            recent_data = filtered_df[filtered_df['year'] == recent_year]
            
            if len(recent_data) > 0:
                top_countries = recent_data.nlargest(5, selected_debt_var)[['country_name', selected_debt_var]]
                
                fig = px.bar(
                    top_countries,
                    x='country_name',
                    y=selected_debt_var,
                    title=f"Top 5 des pays par dette ({recent_year})",
                    labels={'country_name': 'Pays', selected_debt_var: 'Dette (USD)'},
                    color=selected_debt_var,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ==================== ONGLET 2: ANALYSE PAR PAYS ====================
    with tab2:
        st.markdown('<h2 class="sub-header">🇺🇳 Analyse Détailée par Pays</h2>', unsafe_allow_html=True)
        
        if selected_countries:
            selected_country = st.selectbox(
                "Choisir un pays pour l'analyse détaillée",
                selected_countries,
                key="detail_country"
            )
            
            if selected_country:
                country_data = filtered_df[filtered_df['country_name'] == selected_country]
                
                if len(country_data) > 0:
                    # Métriques du pays
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        latest_debt = country_data[selected_debt_var].iloc[-1]
                        st.metric(
                            "Dette Actuelle",
                            f"${latest_debt:,.0f}",
                            help=f"Dette en {country_data['year'].iloc[-1]}"
                        )
                    
                    with col2:
                        if len(country_data) > 1:
                            debt_change = ((country_data[selected_debt_var].iloc[-1] / 
                                          country_data[selected_debt_var].iloc[0]) - 1) * 100
                        else:
                            debt_change = 0
                        st.metric(
                            "Évolution",
                            f"{debt_change:.1f}%",
                            delta=f"{debt_change:.1f}%"
                        )
                    
                    with col3:
                        avg_growth = country_data['SP_POP_GROW'].mean() if 'SP_POP_GROW' in country_data.columns else 0
                        st.metric(
                            "Croissance Démographique",
                            f"{avg_growth:.2f}%",
                            help="Moyenne sur la période"
                        )
                    
                    # Graphiques
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Évolution Temporelle")
                        fig = px.line(
                            country_data,
                            x='year',
                            y=selected_debt_var,
                            title=f'Dette de {selected_country}',
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("##### Corrélations")
                        corr_fig, corr_matrix = analyze_correlations(country_data, selected_country)
                        
                        if corr_fig:
                            st.plotly_chart(corr_fig, use_container_width=True)
                            
                            # Afficher les corrélations les plus fortes
                            if corr_matrix is not None and selected_debt_var in corr_matrix.columns:
                                debt_correlations = corr_matrix[selected_debt_var].drop(selected_debt_var)
                                top_correlations = debt_correlations.abs().nlargest(3)
                                
                                st.markdown("**Corrélations principales avec la dette:**")
                                for var, corr in top_correlations.items():
                                    correlation_value = debt_correlations[var]
                                    st.write(f"- **{var}**: {correlation_value:.3f}")
                        else:
                            st.info("Données insuffisantes pour l'analyse de corrélation")
                    
                    # Données brutes
                    with st.expander("📋 Données détaillées"):
                        st.dataframe(
                            country_data.sort_values('year'),
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.warning(f"Aucune donnée disponible pour {selected_country}")
        else:
            st.warning("Veuillez sélectionner des pays dans la sidebar")
    
    # ==================== ONGLET 3: PRÉDICTIONS ====================

    with tab3:
        st.markdown('<h2 class="sub-header">🔮 Prédictions de la Dette</h2>', unsafe_allow_html=True)
        
        # Avertissement
        st.warning("""
        ⚠️ **Important : Limitations des prédictions**
        
        Les prédictions sont basées sur des tendances historiques et ont des limites :
        1. Ne tiennent pas compte des chocs économiques futurs
        2. Supposent une continuité des politiques actuelles
        3. Les données historiques peuvent être incomplètes
        
        **Utilisez ces prédictions comme indicateurs, non comme certitudes.**
        """)
        
        if selected_countries:
            pred_country = st.selectbox(
                "Sélectionner un pays",
                selected_countries,
                key="prediction_country"
            )
            
            # Afficher les données historiques
            country_history = df[df['country_name'] == pred_country].sort_values('year')
            
            if len(country_history) > 0:
                st.markdown("#### 📊 Données Historiques")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Statistiques de base
                    st.metric(
                        "Années de données",
                        len(country_history),
                        help="Nombre d'années disponibles"
                    )
                
                with col2:
                    latest_value = country_history[selected_debt_var].iloc[-1]
                    st.metric(
                        f"Dette ({country_history['year'].max()})",
                        f"${latest_value:,.0f}",
                        delta_color="inverse"
                    )
                
                # Graphique historique
                fig_hist = px.line(
                    country_history,
                    x='year',
                    y=selected_debt_var,
                    title=f'Évolution historique - {pred_country}',
                    markers=True
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Sélection de la méthode
            st.markdown("#### ⚙️ Paramètres de Prédiction")
            
            method = st.selectbox(
                "Méthode de prédiction",
                [
                    'Ensemble (recommandé)',
                    'Moyenne mobile',
                    'Régression linéaire', 
                    'Croissance moyenne',
                    'Lissage exponentiel'
                ],
                help="""Ensemble combine plusieurs méthodes pour plus de robustesse"""
            )
            
            # Mapping des noms d'affichage aux noms de méthode
            method_map = {
                'Ensemble (recommandé)': 'ensemble',
                'Moyenne mobile': 'moyenne_mobile',
                'Régression linéaire': 'regression_lineaire',
                'Croissance moyenne': 'croissance_moyenne',
                'Lissage exponentiel': 'exponentiel_lisse'
            }
            
            if st.button("🔮 Générer les Prédictions", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyse des tendances historiques..."):
                    # Générer les prédictions
                    predictions = generate_robust_predictions(df, pred_country, selected_debt_var)
                    
                    # Vérifier s'il y a une erreur
                    if 'error' in predictions:
                        st.error(f"❌ {predictions['error']}")
                        
                        # Afficher les données disponibles
                        if 'available_years' in predictions:
                            st.info(f"**Données disponibles :** {predictions['available_years']} années")
                            
                            # Afficher les données brutes
                            with st.expander("📋 Voir les données disponibles"):
                                display_data = country_history[['year', selected_debt_var]].copy()
                                display_data[selected_debt_var] = display_data[selected_debt_var].apply(
                                    lambda x: f"${x:,.0f}"
                                )
                                st.dataframe(display_data, use_container_width=True, hide_index=True)
                    else:
                        st.success("✅ Prédictions générées avec succès!")
                        
                        # Métriques
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Données utilisées",
                                f"{predictions['data_points']} années",
                                help="Nombre d'années historiques utilisées"
                            )
                        
                        with col2:
                            trend_icon = "📈" if predictions['trend'] == 'croissante' else "📉"
                            st.metric(
                                "Tendance",
                                f"{trend_icon} {predictions['trend']}",
                                help="Direction générale des prédictions"
                            )
                        
                        with col3:
                            last_val = predictions['last_actual_value']
                            pred_5yr = predictions['predictions'][-1]
                            growth_5yr = ((pred_5yr - last_val) / abs(last_val)) * 100 if last_val != 0 else 0
                            st.metric(
                                "Croissance sur 5 ans",
                                f"{growth_5yr:.1f}%",
                                delta=f"{growth_5yr:.1f}%"
                            )
                        
                        # Tableau des prédictions
                        st.markdown("#### 📈 Prédictions Détaillées")
                        
                        pred_df = pd.DataFrame({
                            'Année': predictions['future_years'],
                            f'Dette Prédite ({selected_debt_var})': predictions['predictions'],
                            'Variation annuelle': [0] + [
                                ((predictions['predictions'][i] - predictions['predictions'][i-1]) / 
                                abs(predictions['predictions'][i-1]) * 100) 
                                if predictions['predictions'][i-1] != 0 else 0
                                for i in range(1, len(predictions['predictions']))
                            ]
                        })
                        
                        # Mise en forme
                        def format_currency(x):
                            if abs(x) >= 1e9:
                                return f"${x/1e9:.2f}B"
                            elif abs(x) >= 1e6:
                                return f"${x/1e6:.2f}M"
                            else:
                                return f"${x:,.0f}"
                        
                        styled_df = pred_df.style.format({
                            f'Dette Prédite ({selected_debt_var})': format_currency,
                            'Variation annuelle': '{:.1f}%'
                        })
                        
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        # Graphique combiné
                        st.markdown("#### 📊 Visualisation des Prédictions")
                        
                        fig = go.Figure()
                        
                        # Historique
                        fig.add_trace(go.Scatter(
                            x=country_history['year'],
                            y=country_history[selected_debt_var],
                            mode='lines+markers',
                            name='Historique',
                            line=dict(color='#1f77b4', width=3),
                            marker=dict(size=8),
                            hovertemplate='<b>Historique</b><br>Année: %{x}<br>Dette: %{y:,.0f}<extra></extra>'
                        ))
                        
                        # Prédictions
                        fig.add_trace(go.Scatter(
                            x=predictions['future_years'],
                            y=predictions['predictions'],
                            mode='lines+markers',
                            name='Prédictions',
                            line=dict(color='#ff7f0e', width=3, dash='dash'),
                            marker=dict(size=10, symbol='diamond'),
                            hovertemplate='<b>Prédiction</b><br>Année: %{x}<br>Dette: %{y:,.0f}<extra></extra>'
                        ))
                        
                        # Point de jonction
                        fig.add_trace(go.Scatter(
                            x=[predictions['last_actual_year']],
                            y=[predictions['last_actual_value']],
                            mode='markers',
                            name='Dernière valeur connue',
                            marker=dict(size=12, color='red', symbol='star'),
                            hovertemplate='<b>Dernière valeur</b><br>Année: %{x}<br>Dette: %{y:,.0f}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title=f'Prédictions de la Dette - {pred_country}',
                            xaxis_title='Année',
                            yaxis_title='Dette (USD)',
                            hovermode='x unified',
                            template='plotly_white',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            height=500
                        )
                        
                        # Ajuster l'échelle Y si nécessaire
                        all_values = list(country_history[selected_debt_var]) + predictions['predictions']
                        if max(all_values) / min([v for v in all_values if v > 0]) > 100:
                            fig.update_yaxes(type="log")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Si méthode ensemble, montrer les différentes prédictions
                        if method == 'Ensemble (recommandé)' and 'all_predictions' in predictions:
                            st.markdown("#### 🔍 Détail des Méthodes d'Ensemble")
                            
                            methods_df = pd.DataFrame(predictions['all_predictions'])
                            methods_df['Année'] = predictions['future_years']
                            
                            fig_methods = go.Figure()
                            
                            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                            
                            for i, (method_name, values) in enumerate(predictions['all_predictions'].items()):
                                fig_methods.add_trace(go.Scatter(
                                    x=predictions['future_years'],
                                    y=values,
                                    mode='lines',
                                    name=method_name.replace('_', ' ').title(),
                                    line=dict(color=colors[i % len(colors)], width=2)
                                ))
                            
                            # Moyenne ensemble
                            fig_methods.add_trace(go.Scatter(
                                x=predictions['future_years'],
                                y=predictions['predictions'],
                                mode='lines+markers',
                                name='Moyenne Ensemble',
                                line=dict(color='black', width=4),
                                marker=dict(size=8)
                            ))
                            
                            fig_methods.update_layout(
                                title='Comparaison des Méthodes de Prédiction',
                                xaxis_title='Année',
                                yaxis_title='Dette Prédite (USD)',
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_methods, use_container_width=True)
                        
                        # Téléchargement
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Export CSV
                            csv = pred_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Télécharger les prédictions (CSV)",
                                data=csv,
                                file_name=f"predictions_{pred_country}_{selected_debt_var}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Rapport texte
                            report_text = f"""
                            RAPPORT DE PRÉDICTION - DETTE
                            Pays: {pred_country}
                            Variable: {selected_debt_var}
                            Méthode: {predictions['method']}
                            Date de génération: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
                            
                            DONNÉES UTILISÉES:
                            - Période historique: {country_history['year'].min()} - {predictions['last_actual_year']}
                            - Années de données: {predictions['data_points']}
                            - Dernière valeur: ${predictions['last_actual_value']:,.0f}
                            
                            PRÉDICTIONS (5 ans):
                            """
                            
                            for year, pred in zip(predictions['future_years'], predictions['predictions']):
                                report_text += f"- {year}: ${pred:,.0f}\n"
                            
                            report_text += f"""
                            
                            ANALYSE:
                            - Tendance: {predictions['trend']}
                            - Croissance totale projetée: {growth_5yr:.1f}%
                            - Valeur projetée dans 5 ans: ${predictions['predictions'][-1]:,.0f}
                            
                            NOTES:
                            Ces prédictions sont basées sur des tendances historiques.
                            Elles ne prennent pas en compte les événements futurs imprévus.
                            """
                            
                            st.download_button(
                                label="📄 Télécharger le rapport (TXT)",
                                data=report_text,
                                file_name=f"rapport_{pred_country}_dette.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
        
        # Section éducative
        with st.expander("📚 Comment fonctionnent les prédictions?"):
            st.markdown("""
            **Méthodes utilisées:**
            
            1. **Moyenne Mobile**:
               - Calcule la moyenne des dernières années
               - Lisse les variations à court terme
               - Bon pour les séries stables
            
            2. **Régression Linéaire**:
               - Trouve la ligne droite qui s'ajuste le mieux aux données
               - Extrapole la tendance linéaire
               - Simple mais efficace
            
            3. **Croissance Moyenne**:
               - Calcule le taux de croissance historique moyen
               - Applique ce taux aux années futures
               - Conserve la dynamique de croissance
            
            4. **Lissage Exponentiel**:
               - Donne plus de poids aux données récentes
               - Réagit mieux aux changements récents
               - Bon pour les séries avec tendance
            
            5. **Ensemble (recommandé)**:
               - Combine les 4 méthodes
               - Moyenne des prédictions
               - Réduit les erreurs individuelles
               - Plus robuste aux variations
            """)
    # ==================== ONGLET 4: COMPARAISONS ====================
    with tab4:
        st.markdown('<h2 class="sub-header">📊 Analyse Comparative</h2>', unsafe_allow_html=True)
        
        if len(selected_countries) >= 2:
            # Heatmap comparative
            st.markdown("#### 📍 Comparaison des Dettes par Année")
            
            pivot_data = filtered_df.pivot_table(
                index='country_name',
                columns='year',
                values=selected_debt_var,
                aggfunc='mean'
            ).fillna(0)
            
            if len(pivot_data) > 0:
                fig = px.imshow(
                    pivot_data,
                    title='Heatmap des Dettes par Pays et Année',
                    color_continuous_scale='Viridis',
                    labels=dict(x="Année", y="Pays", color="Dette"),
                    aspect="auto"
                )
                
                fig.update_layout(
                    xaxis=dict(tickangle=45),
                    height=400 + len(pivot_data) * 20
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Comparaison des évolutions
            st.markdown("#### 📈 Évolution Comparative")
            
            comparison_fig = go.Figure()
            
            for country in selected_countries[:5]:  # Limiter à 5 pays pour la lisibilité
                country_data = filtered_df[filtered_df['country_name'] == country].sort_values('year')
                if len(country_data) > 0:
                    # Normalisation pour comparer les tendances
                    if country_data[selected_debt_var].max() > 0:
                        normalized = (country_data[selected_debt_var] / country_data[selected_debt_var].max()) * 100
                    else:
                        normalized = country_data[selected_debt_var]
                    
                    comparison_fig.add_trace(go.Scatter(
                        x=country_data['year'],
                        y=normalized,
                        mode='lines',
                        name=country,
                        hovertemplate=f"{country}<br>Année: %{{x}}<br>Dette (normalisée): %{{y:.1f}}%<extra></extra>"
                    ))
            
            comparison_fig.update_layout(
                title='Évolution Comparative (normalisée à 100%)',
                xaxis_title='Année',
                yaxis_title='Dette (normalisée)',
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Tableau comparatif
            st.markdown("#### 📋 Données Comparatives")
            
            if len(filtered_df) > 0:
                latest_year = filtered_df['year'].max()
                latest_data = filtered_df[filtered_df['year'] == latest_year]
                
                if len(latest_data) > 0:
                    comparison_df = latest_data[['country_name', selected_debt_var]].sort_values(
                        selected_debt_var, ascending=False
                    )
                    
                    # Ajouter le classement
                    comparison_df['Rang'] = range(1, len(comparison_df) + 1)
                    
                    st.dataframe(
                        comparison_df.style.format({
                            selected_debt_var: '{:,.0f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
        else:
            st.info("Sélectionnez au moins 2 pays pour les comparaisons")
    
    # ==================== ONGLET 5: RECOMMANDATIONS ====================
    with tab5:
        st.markdown('<h2 class="sub-header">📋 Analyse de Risque et Recommandations</h2>', unsafe_allow_html=True)
        
        # Analyse des risques
        risk_classifier = DebtRiskClassifier()
        risk_df = risk_classifier.calculate_debt_indicators(filtered_df)
        
        if len(risk_df) > 0:
            # Carte des risques
            st.markdown("#### 🚨 Niveaux de Risque par Pays")
            
            # Classes CSS pour les risques
            risk_colors = {
                'Faible': '🟢',
                'Modéré': '🟡',
                'Élevé': '🟠',
                'Critique': '🔴'
            }
            
            risk_css = {
                'Faible': 'risk-low',
                'Modéré': 'risk-medium',
                'Élevé': 'risk-high',
                'Critique': 'risk-critical'
            }
            
            for country in selected_countries:
                if country in risk_df.index:
                    risk_data = risk_df.loc[country]
                    risk_level = risk_data['risk_level']
                    
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown(f"<h3>{risk_colors[risk_level]} {country}</h3>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <p><strong>Ratio Dette/PIB estimé:</strong> 
                                <span class="{risk_css[risk_level]}">{risk_data['debt_gdp_ratio']:.1f}%</span></p>
                                <p><strong>Croissance annuelle de la dette:</strong> 
                                <span class="{risk_css[risk_level]}">{risk_data['debt_growth']:.1f}%</span></p>
                                <p><strong>Niveau de risque:</strong> 
                                <span class="{risk_css[risk_level]}"><strong>{risk_level}</strong></span></p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Recommandations spécifiques
                    with st.expander(f"Recommandations pour {country}"):
                        if risk_level == 'Faible':
                            st.success("""
                            **✅ Situation favorable:**
                            - Maintenir la discipline fiscale
                            - Profiter des taux bas pour des investissements stratégiques
                            - Diversifier les sources de financement
                            """)
                        elif risk_level == 'Modéré':
                            st.warning("""
                            **⚠️ Vigilance requise:**
                            - Surveiller étroitement la croissance de la dette
                            - Renforcer les recettes fiscales
                            - Prioriser les investissements productifs
                            - Éviter l'endettement à court terme
                            """)
                        elif risk_level == 'Élevé':
                            st.error("""
                            **🚨 Action corrective nécessaire:**
                            - Mettre en place un plan de consolidation fiscale
                            - Renégocier les termes de la dette si possible
                            - Réduire les dépenses non essentielles
                            - Rechercher des financements concessionnels
                            - Augmenter les réserves de change
                            """)
                        else:  # Critique
                            st.error("""
                            **🔴 Situation critique:**
                            - Demander une restructuration de la dette
                            - Solliciter l'assistance du FMI/BM
                            - Mettre en œuvre des réformes structurelles urgentes
                            - Geler les nouveaux emprunts non essentiels
                            - Prioriser le service de la dette existante
                            """)
                    
                    st.divider()
            
            # Recommandations générales
            st.markdown("#### 💡 Recommandations Générales")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **📊 Transparence des données:**
                - Publier régulièrement les statistiques de dette
                - Standardiser les rapports selon les normes internationales
                - Créer des observatoires nationaux de la dette
                """)
            
            with col2:
                st.markdown("""
                **🌍 Coopération régionale:**
                - Partager les meilleures pratiques
                - Négocier collectivement avec les créanciers
                - Créer des fonds régionaux de stabilisation
                """)
            
            with col3:
                st.markdown("""
                **🔮 Prospective stratégique:**
                - Développer des modèles de simulation
                - Anticiper les chocs externes
                - Diversifier l'économie
                - Investir dans le capital humain
                """)
            
            # Téléchargement du rapport
            st.markdown("---")
            st.markdown("#### 📄 Export des Analyses")
            
            if st.button("📊 Générer un Rapport Complet"):
                with st.spinner("Génération du rapport..."):
                    # Créer un rapport simplifié
                    report_data = {
                        'Période_analyse': f"{selected_years[0]}-{selected_years[1]}",
                        'Pays_analysés': len(selected_countries),
                        'Dette_moyenne': f"${avg_debt:,.0f}",
                        'Date_génération': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    report_df = pd.DataFrame([report_data])
                    
                    csv = report_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger le Rapport (CSV)",
                        data=csv,
                        file_name="rapport_dette_afrique.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("Données insuffisantes pour l'analyse de risque")
    
    # ==================== PIED DE PAGE ====================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>📊 <strong>Dashboard d'Analyse de la Dette Africaine</strong> | 
    Données: World Bank & Autres Sources | 
    Dernière mise à jour: 2024</p>
    <p><em>Cet outil est destiné à l'analyse et ne constitue pas un conseil financier</em></p>
    </div>
    """, unsafe_allow_html=True)

# ==================== EXÉCUTION ====================
if __name__ == "__main__":
    main()