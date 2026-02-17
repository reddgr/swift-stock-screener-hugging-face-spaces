import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import gamma
import json

class FrontDatasetHandler:
    def __init__(self, maestro: pd.DataFrame=None, precios_cierre: pd.DataFrame=None, app_dataset: pd.DataFrame=None, 
                 json_path: str = None, pickle_path: str = None, ignore_columns: list = None, numeric_columns: list = None):
        self.maestro = maestro
        self.app_dataset = app_dataset # Dataframe preprocesado para la app
        self.pickle_path = pickle_path
        # Extraemos los ficheros JSON para la creación del dataset de la app si no se ha pasado como argumento:
        if self.app_dataset is None and json_path is not None:
            with open(os.path.join(json_path, "ignore_columns.json"), "r") as f:
                self.ignore_columns = json.load(f)['ignore_columns']
            print(f"ignore_columns: {self.ignore_columns}")
            with open(os.path.join(json_path, "numeric_columns.json"), "r") as f:
                self.numeric_columns = json.load(f)['numeric_columns']
            print(f"numeric_columns: {self.numeric_columns}")
            with open(os.path.join(json_path, "app_column_config.json"), "rb") as f:
                self.app_dataset_cols  = json.load(f)['app_dataset_cols']
            print(f"app_dataset_cols: {self.app_dataset_cols}")

            with open(os.path.join(json_path, "cat_to_num_maps.json"), "r") as f:
                num_maps = json.load(f)
            self.sector_num_map = num_maps['sector_num_map']
            self.industry_num_map = num_maps['industry_num_map']
        else:
            self.ignore_columns = ignore_columns
            self.numeric_columns = numeric_columns
            print(f"ignore_columns: {self.ignore_columns}")
            print(f"numeric_columns: {self.numeric_columns}")

        self.norm_columns = None
        if maestro is not None:
            maestro.drop(columns=self.ignore_columns, inplace=True, errors='ignore')
        self.precios_cierre = precios_cierre # Sólo necesario cuando se requiere preprocesar el dataset para la app
        self.rend_diario_log = None
        self.precios_cierre_fh = None
        self.rendimientos_y_volatilidad = None
        self.mapeos_var_categoricas = None
        self.activos_descartados = []
        self.quantile_scaler = None

    def filtra_y_homogeneiza(self, n_dias=366, n_dias_descartar=1, min_dias=100):
        if self.precios_cierre.index.name != 'date':
            self.precios_cierre.set_index('date', inplace=True)
        self.precios_cierre.columns.name = 'ticker'
        end_date = self.precios_cierre.index.max()
        start_date = end_date - pd.Timedelta(days=n_dias)
        # If start_date is not in the index, find the nearest earlier date
        if start_date not in self.precios_cierre.index:
            earlier_dates = self.precios_cierre.index[self.precios_cierre.index < start_date]
            if len(earlier_dates) > 0:
                start_date = earlier_dates.max()
            else:
                start_date = self.precios_cierre.index.min()
        
        # Filtrar datos dentro del rango de fechas 
        precios_cierre_fh = self.precios_cierre.loc[start_date:end_date].copy()
        
        # Descartar los últimos n_dias_descartar
        if n_dias_descartar > 0:
            dates_to_drop = precios_cierre_fh.index.sort_values(ascending=False)[:n_dias_descartar]
            precios_cierre_fh.drop(dates_to_drop, inplace=True)
        
        precios_cierre_fh.ffill(axis=0, inplace=True) # Se rellenan los datos vacíos con el dato del día anterior
        self.activos_descartados = precios_cierre_fh.columns[precios_cierre_fh.notna().sum(axis=0) < min_dias].tolist()
        precios_cierre_fh.drop(columns=self.activos_descartados, inplace=True)
        self.precios_cierre = precios_cierre_fh
        return
    
    def calcula_rendimientos_y_volatilidad(self, n_dias=365, umbral_max=0.3, umbral_min=-0.3):
        end_date = self.precios_cierre.index.max()
        print(f"Última fecha: {end_date}")
        print("Primera fecha: ",self.precios_cierre.index.min())
        start_date = end_date - pd.Timedelta(days=n_dias)
        # Si no hay cotizaciones para la fecha de inicio calculada (ej. fin de semana), se cambia por la fecha más cercana
        if start_date not in self.precios_cierre.index:
            earlier_dates = self.precios_cierre.index[self.precios_cierre.index < start_date]
            if len(earlier_dates) > 0:
                start_date = earlier_dates.max()
            else:
                start_date = self.precios_cierre.index.min()
                
        _df_rend_y_vol = self.precios_cierre.loc[start_date:end_date].copy()
        
        _df_rend_y_vol.dropna(how='all', inplace=True) #####
        # Reemplazar valores cero y negativos (errores de formato) por el siguiente valor más pequeño positivo
        _df_rend_y_vol[_df_rend_y_vol <= 0] = np.nextafter(0, 1)
        if self.activos_descartados:
            _df_rend_y_vol = _df_rend_y_vol.drop(columns=[col for col in self.activos_descartados if col in _df_rend_y_vol.columns])   
        if len(_df_rend_y_vol) == 0:
            raise ValueError(f"No hay datos disponibles en el rango de {n_dias} días")
        
        
        _rend_diario_log = np.log(_df_rend_y_vol).diff()
        _rend_diario_log = _rend_diario_log.iloc[1:] # Eliminar la primera fila
        # _rend_diario_log.dropna(how='all', inplace=True)  
        print(f'Datos rentabilidad ({n_dias} días) con outliers: {_rend_diario_log.shape}')
        # Identificar activos a descartar (outliers)
        _activos_outliers = _rend_diario_log.columns[((_rend_diario_log > umbral_max) | (_rend_diario_log < umbral_min)).any()].tolist()
        self.activos_descartados.extend([asset for asset in _activos_outliers if asset not in self.activos_descartados])
        # Descartar activos con rentabilidades atípicas
        _rend_diario_log = _rend_diario_log.loc[:, ~((_rend_diario_log > umbral_max) | (_rend_diario_log < umbral_min)).any()]
        print(f'Datos rentabilidad sin outliers: {_rend_diario_log.shape}')
        
        self.rend_diario_log = _rend_diario_log.copy()
        
        # Inicializar rendimientos_y_volatilidad si no existe
        if self.rendimientos_y_volatilidad is None:
            self.rendimientos_y_volatilidad = pd.DataFrame(columns=_rend_diario_log.columns)
            # print(f'INIT: Tabla rendimientos {n_dias}: {self.rendimientos_y_volatilidad.shape}')
        else:
            # Mantener solo los activos que están en _rend_diario_log
            self.rendimientos_y_volatilidad = self.rendimientos_y_volatilidad.loc[:, _rend_diario_log.columns]
            # print(f'Tabla rendimientos {n_dias}: {self.rendimientos_y_volatilidad.shape}')
            
        # Añadir nuevas columnas para el n_dias actual
        self.rendimientos_y_volatilidad.loc[f'ret_log_{n_dias}'] = np.sum(_rend_diario_log, axis=0)
        self.rendimientos_y_volatilidad.loc[f'ret_{n_dias}'] = (_df_rend_y_vol.ffill().bfill().iloc[-1] / _df_rend_y_vol.ffill().bfill().iloc[0]) - 1
        self.rendimientos_y_volatilidad.loc[f'vol_{n_dias}'] = _rend_diario_log.var()**0.5
 
        return
    
    def cruza_maestro(self):
        _rets_y_vol_maestro = self.rendimientos_y_volatilidad.T.reset_index().copy()
        _columns_to_merge = [col for col in _rets_y_vol_maestro.columns if col not in self.maestro.columns]     
        if len(_columns_to_merge) > 0:
            _maestro_v2 = self.maestro.merge(_rets_y_vol_maestro, left_on='ticker', right_on='ticker')
            _maestro_v2 = _maestro_v2.replace([float('inf'), float('-inf')], np.nan)
            self.maestro = _maestro_v2
        else:
            raise ValueError("No hay nuevas columnas para cruzar con el dataframe maestro")
        return
    
    def _cat_to_num_(self, df, cat, pre_map=None):
        """
        Transforma una columna categórica en un DataFrame a valores numéricos asignando un número entero a cada categoría.
        Si no se proporciona un mapeo (`pre_map`), asigna 0 a la categoría más frecuente, 1 a la siguiente más frecuente, y así sucesivamente.
        Si se proporciona un mapeo (`pre_map`), utiliza ese mapeo para la conversión.
        Parámetros
        ----------
        df : pandas.DataFrame
            DataFrame que contiene la columna categórica a transformar.
        cat : str
            Nombre de la columna categórica a transformar.
        pre_map : dict, opcional
            Diccionario que mapea cada categoría a un valor numérico. Si no se proporciona, el mapeo se genera automáticamente.
        Devuelve
        --------
        pandas.DataFrame
            DataFrame con dos columnas: la columna categórica original y una columna con los valores numéricos asignados.
        """
        if not pre_map:
            pivot = pd.pivot_table(df, index=[cat], aggfunc='size')
            df_sorted = pivot.sort_values(ascending=False).reset_index(name='count')
            df_sorted[cat + '_num'] = range(len(df_sorted))
        else:
            df_sorted = pd.DataFrame({cat: list(pre_map.keys()), cat + '_num': list(pre_map.values())})
        return df_sorted
    
    def var_categorica_a_numerica(self, cat_cols):
        for col in cat_cols:
            if col == 'sectorDisp':
                globals()[f"pt_{col}"] = self._cat_to_num_(self.maestro, col, self.sector_num_map)
            elif col == 'industryDisp':
                globals()[f"pt_{col}"] = self._cat_to_num_(self.maestro, col, self.industry_num_map)
            else:
                globals()[f"pt_{col}"] = self._cat_to_num_(self.maestro, col)  # Creamos un dataframe con el mapeo de cada variable categórica por frecuencia
        self.mapeos_var_categoricas = [globals()[f"pt_{col}"] for col in cat_cols] # Lista de dataframes con los mapeos de cada una de las variables categóricas

        _maestro = self.maestro.copy()
        for col, pt in zip(cat_cols, self.mapeos_var_categoricas):
            _maestro[col] = _maestro[col].astype(str)
            pt[col] = pt[col].astype(str)
            # Creamos un diccionario con cada variable categórica y su equivalente numérico
            mapping_dict = dict(zip(pt[col], pt[col + '_num']))
            _maestro[col + '_num'] = _maestro[col].map(mapping_dict)
            _maestro[col + '_num'] = pd.to_numeric(_maestro[col + '_num'], errors='coerce')

        self.maestro = _maestro
        return
    
    def normaliza_por_cuantiles(self):
        maestro_copy = self.maestro.copy()
        numeric_columns = maestro_copy.select_dtypes(include=np.number).columns
        self.quantile_scaler = QuantileTransformer(output_distribution='uniform')
        variables_numericas = [col for col in numeric_columns if not col.endswith('_norm')]
        all_na_cols = [col for col in variables_numericas if maestro_copy[col].isna().all()]
        variables_numericas = [col for col in variables_numericas if col not in all_na_cols]
        self.norm_columns = ['{}_norm'.format(var) for var in variables_numericas]
        maestro_copy[self.norm_columns] = self.quantile_scaler.fit_transform(maestro_copy[variables_numericas])
        maestro_copy[self.norm_columns] = maestro_copy[self.norm_columns].clip(0, 1)
        self.maestro = maestro_copy
        return
    
    def var_estandar_z(self):
        maestro_copy = self.maestro.copy()
        numeric_columns = maestro_copy.select_dtypes(include=np.number).columns
        variables_numericas = [col for col in numeric_columns if not col.endswith('_std')]
        variables_num_std = ['{}_std'.format(var) for var in variables_numericas]

        def estandarizar(x): 
            # Estandariza el valor z, restando la media y dividiendo por la desviación estándar
            mean_val = x.mean()
            std_val = x.std()
            if pd.isna(std_val) or std_val == 0:
                return pd.Series(0.0, index=x.index, name=x.name)
            else:
                normalized_series = (x - mean_val) / std_val
                return normalized_series.fillna(0.0)

        normalized_data = maestro_copy[variables_numericas].apply(estandarizar, axis=0)
        maestro_copy[variables_num_std] = normalized_data
        self.maestro = maestro_copy
        return
    
    def configura_distr_prob(self, shape, loc, scale, max_dist, precision_cdf):
        x = np.linspace(0, max_dist, num=precision_cdf)
        y_pdf = gamma.pdf(x, shape, loc, scale )
        y_cdf = gamma.cdf(x, shape, loc, scale )
        return y_pdf, y_cdf
    
    def calculos_y_ajustes_dataset_activos(self):
        maestro_copy = self.maestro.copy()
        # Conversiones a formato numérico de columnas que dan problemas
        for column in self.numeric_columns:
            if column in maestro_copy.columns:
                maestro_copy[column] = pd.to_numeric(maestro_copy[column], errors='coerce') 
                # print(f"Columna {column} convertida a {maestro_copy[column].dtype}")
        # Estandarización de los diferentes tipos de NaN
        # maestro_copy = maestro_copy.replace([None, np.nan, np.inf, -np.inf], pd.NA)
        # Antigüedad del fondo en años:
        if self.precios_cierre is not None and not self.precios_cierre.index.empty:
            _most_recent_date = self.precios_cierre.index.max().date()
        else:
            _most_recent_date = pd.Timestamp.today().date()
        # maestro_copy['firstTradeDateMilliseconds'] = pd.to_datetime(maestro_copy['firstTradeDateMilliseconds']).dt.date
        maestro_copy['firstTradeDateMilliseconds'] = pd.to_datetime(maestro_copy['firstTradeDateMilliseconds'], unit='ms', errors='coerce').dt.date 
        maestro_copy['asset_age'] = maestro_copy['firstTradeDateMilliseconds'].apply(
            lambda x: ((_most_recent_date - x).days / 365) if pd.notnull(x) and hasattr(x, 'day') else 0
        ).astype(int)
        outlier_thresholds = {
            'beta': (-100, 100),
            'dividendYield': (-1,100),
            'fiveYearAvgDividendYield': (-1,100),
            'trailingAnnualDividendYield': (-1,100),
            'quickRatio': (-1, 500),
            'currentRatio': (-1, 500),
            'ebitda': (-1e12, 1e12),
            'grossProfits': (-1e12, 1e12),
        }
        for column, (lower_bound, upper_bound) in outlier_thresholds.items():
            maestro_copy.loc[(maestro_copy[column] < lower_bound) | (maestro_copy[column] > upper_bound), column] = pd.NA
        self.maestro = maestro_copy.copy()
        return
    
    def filtra_df_activos(self, df, isin_target, filtros, debug=False):
        '''
        LEGACY
        Devuelve un dataframe filtrado, sin alterar el orden, eliminando características no deseadas, para usar en aplicación de búsqueda de activos sustitutivos.
        Las características y valores a filtrar son las de un fondo objetivo dado por su isin. 
        Por ejemplo, si clean_share es False en filtros, el dataframe final no incluirá más activos con el mismo valor de clean_share que el ISIN objetivo
        Argumentos:
            df (pandas.core.frame.DataFrame): Dataframe maestro de activos
            isin_target (str): ISIN del fondo objetivo
            # caracteristicas (list): Lista de str con los nombres de las características
            filtros (dict): Diccionario donde las claves son las características y los valores son True si se quiere conservar
            debug (bool, optional): Muestra información de depuración. Por defecto False.
        Resultado:
            df_filt (pandas.core.frame.DataFrame): Dataframe filtrado
        '''
        # fondo_target = df[df['isin'] == isin_target].iloc[0]
        fondo_target = df[df['ticker'] == isin_target].iloc[0]
        if debug: print(f'Tamaño inicial: {df.shape}')
        
        car_numericas = ['ret_365', 'vol_365', 'marketCap', 'asset_age']
        
        # for feature in caracteristicas[2:]:
        for feature in list(filtros.keys()):
            value = fondo_target[feature]
            if debug: print(f'{feature} = {value}')
            
            # Verificar si esta característica debe ser filtrada
            if feature in filtros and not filtros[feature]:
                if debug: print(f'FILTRO: {feature} != {value}')
                df = df[df[feature] != value]
                
                # Aplicar filtros adicionales para variables numéricas
                if feature in car_numericas:
                    if feature == 'ret_365':
                        if debug: print(f'FILTRO NUMÉRICO: {feature} > {value}')
                        df = df[df[feature] > value]
                    elif feature == 'vol_365':
                        if debug: print(f'FILTRO NUMÉRICO: {feature} < {value}')
                        df = df[df[feature] < value]
                    elif feature == 'asset_age':
                        if debug: print(f'FILTRO NUMÉRICO: {feature} > {value}')
                        df = df[df[feature] > value]
                    elif feature == 'marketCap':
                        if debug: print(f'FILTRO NUMÉRICO: {feature} > {value}')
                        df = df[df[feature] < value]
        
        df_filt = df
        if debug: print(f'Tamaño final: {df_filt.shape}')
        return df_filt
    
    def calcula_ind_sust (self, dist, y_cdf, precision_cdf, max_dist):
        try:
            idx = int((precision_cdf / max_dist) * dist)
            idx = min(idx, len(y_cdf) - 1)
            norm_dist = y_cdf[idx]
            ind_sust = max(0.0, 1.0 - norm_dist)
        except IndexError:
            ind_sust = 0
        return ind_sust
    

    def vecinos_cercanos(self, df, variables_busq, caracteristicas, target_ticker, y_cdf, precision_cdf, max_dist, n_neighbors, filtros):
        if target_ticker not in df['ticker'].values:
            return f"Error: '{target_ticker}' no encontrado en la base de datos"
        target_row = df[df['ticker'] == target_ticker]
        if ~target_row.index.isin(df.index):
            df = pd.concat([df, target_row], ignore_index=True)
        # print(f'DF original: {df.shape}')
        X = df[variables_busq]
        model = NearestNeighbors(n_neighbors=n_neighbors) ##### probar con más y filtrar después #######
        model.fit(X)
        target_row = df[df['ticker'] == target_ticker][variables_busq]
        # model.kneighbors devuelve dos arrays bidimensionales con los vecinos más cercanos y sus distancias:
        distances, indices = model.kneighbors(target_row)
        # combined_columns = list(set(caracteristicas + variables_busq))
        neighbors_df = df.iloc[indices[0]][caracteristicas]
        neighbors_df['distance'] = distances[0]
        ind_sust = np.array([self.calcula_ind_sust(dist, y_cdf, precision_cdf, max_dist) for dist in distances[0]])
      
        neighbors_df['ind_sust'] = ind_sust
        neighbors_df = neighbors_df.sort_values(by='distance', ascending=True)  
        target_row = neighbors_df[neighbors_df['ticker'] == target_ticker]
        
        # Aplicamos los filtros de exclusión:
        ### Código pendiente de eliminar/modificar (legado de la aplicación de fondos)
        neighbors_df = self.filtra_df_activos (df = neighbors_df, isin_target = target_ticker, filtros = filtros)
        ####################

        # Recupera el activo seleccionado en caso de haber hecho filtros, devolviéndolo a la primera posición del dataframe:
        if ~target_row.index.isin(neighbors_df.index):
            neighbors_df = pd.concat([pd.DataFrame(target_row), neighbors_df], ignore_index=True)
        # print(f'DF filtrado: {neighbors_df.shape}')
        # Ponemos el ticker como índice:
        neighbors_df.set_index('ticker', inplace = True)
        return neighbors_df
    
    def format_large_number(self, n, decimals=2):
        if n >= 1e12:
            return f'{n / 1e12:.{decimals}f} T'
        elif n >= 1e9:
            return f'{n / 1e9:.{decimals}f} B'
        elif n >= 1e6:
            return f'{n / 1e6:.{decimals}f} M'
        else:
            return str(n)
        
    def trae_embeddings_desde_pkl(self, embeddings_df_file_name='df_with_embeddings.pkl', embeddings_col_name='embeddings'):
        embeddings_df = pd.read_pickle(os.path.join(self.pickle_path, embeddings_df_file_name))
        self.maestro = self.maestro.merge(
            embeddings_df[['ticker', embeddings_col_name]],
            on='ticker',
            how='left'
        )
        print(f"Agregados embeddings {self.maestro.shape}")
        return
        
    def procesa_app_dataset(self, periodo=366, n_dias_descartar=1, min_dias=250, umbrales_rend=(-0.3, +0.3), periodos_metricas=[60, 365],
                        cat_cols = ['industryDisp', 'sectorDisp', 'country', 'city', 'exchange', 'financialCurrency', 'quoteType'],
                        embeddings_df_file_name='df_with_embeddings.pkl', embeddings_col_name='embeddings'):
        if self.app_dataset is not None:
            print("app_dataset already exists, skipping processing")
            return

        self.filtra_y_homogeneiza(n_dias=periodo, n_dias_descartar=n_dias_descartar, min_dias=min_dias)
        
        for periodo_metricas in periodos_metricas: 
            self.calcula_rendimientos_y_volatilidad(n_dias=periodo_metricas, umbral_max=umbrales_rend[1], umbral_min=umbrales_rend[0])
        self.cruza_maestro()
        self.var_categorica_a_numerica(cat_cols)
        
        self.calculos_y_ajustes_dataset_activos()
        self.normaliza_por_cuantiles()
        self.trae_embeddings_desde_pkl(embeddings_df_file_name=embeddings_df_file_name, embeddings_col_name=embeddings_col_name)
        app_dataset = self.maestro.copy()
        app_dataset = app_dataset.fillna({col: 0.5 for col in self.norm_columns})
        # Filtrado final de columnas para reducir el dataset:
        self.app_dataset = app_dataset[self.app_dataset_cols].copy()
        print(f"app_dataset preparado: {self.app_dataset.shape}")
        return