from fastapi import FastAPI,HTTPException
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


app = FastAPI()

def cargar_csv(nombre_archivo):
    try:
        # Obtener la ruta completa al archivo CSV
        ruta_archivo = os.path.join(os.path.dirname(__file__), 'DataSets', nombre_archivo)
        
        # Imprimir la ruta para verificar
        print(f'Ruta del archivo: {ruta_archivo}')
        return pd.read_csv(ruta_archivo, encoding="UTF-8", sep=",")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f'Archivo {nombre_archivo} no encontrado.')

def peliculas_estrenadas_por_dia(nombre_dia):
    try:
        df = cargar_csv('movies_dataset.csv')  # Cargar el archivo de películas
        
        nombre_dia = nombre_dia.lower()
        
        # Validar que el día ingresado sea válido
        dias_semana = df['release_day_of_week'].str.lower().unique()
        if nombre_dia not in dias_semana:
            raise HTTPException(status_code=400, detail=f'Error: "{nombre_dia}" no es un día válido. Por favor, ingrese un día válido en español.')

        # Filtrar las películas estrenadas en el día especificado
        count = df[df['release_day_of_week'].str.lower() == nombre_dia].shape[0]
        
        return {'message': f'{count} películas fueron estrenadas el día {nombre_dia}'}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def peliculas_estrenadas_por_mes(nombre_mes):
    try:
        df = cargar_csv('movies_dataset.csv') 
        nombre_mes = nombre_mes.lower()
        # Validar que el mes ingresado sea válido
        meses_del_año = [
            'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 
            'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'
        ]
        
        if nombre_mes not in meses_del_año:
            raise HTTPException(status_code=400, detail=f'Error: "{nombre_mes}" no es un mes válido. Por favor, ingrese un mes válido en español.')

        # Filtrar las películas estrenadas en el mes especificado
        count = df[df['release_month'].str.lower() == nombre_mes].shape[0]
        
        return {'message': f'{count} películas fueron estrenadas en el mes de {nombre_mes}'}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def informacion_pelicula(nombre_pelicula):
    try:
        df = cargar_csv('movies_dataset.csv') 
        nombre_pelicula = nombre_pelicula.lower()
        peliculas = df[df['title'].str.lower() == nombre_pelicula]
        if len(peliculas) == 0:
            return {'message': f'No se encontró información para la película "{nombre_pelicula}".'}
        
        resultados = []
        for index, row in peliculas.iterrows():
            titulo = row['title']
            popularidad = row['popularity']
            año_estreno = row['release_year']
            resultados.append({
                'message': f"La película '{titulo}' fue estrenada en el año {año_estreno} con un score de popularidad de {popularidad}"
            })
        
        return resultados
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def informacion_pelicula_votaciones(nombre_pelicula):
    try:
        df = cargar_csv('movies_dataset.csv')
        nombre_pelicula = nombre_pelicula.lower()
        peliculas = df[df['title'].str.lower() == nombre_pelicula]
        
        if len(peliculas) == 0:
            return {'message': f'No se encontró información para la película "{nombre_pelicula}".'}
        
        resultados = []
        for index, row in peliculas.iterrows():
            titulo = row['title']
            cantidad_votos = row['vote_count']
            promedio_votaciones = row['vote_average']
            año_estreno = row['release_year']
            
            if cantidad_votos >= 2000:
                resultados.append({
                    'message': f"La película {titulo} fue estrenada en el año {año_estreno}. La misma cuenta con un total de {cantidad_votos} valoraciones, con un promedio de {promedio_votaciones:.2f}."
                })
            else:
                resultados.append({
                    'message': f"La película {titulo} que fue estrenada en el año {año_estreno} no tiene más de 2000 votos, por lo cual no se devuelve ningún valor de esa película."
                })
        
        return resultados
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def informacion_actores_retorno(actor_nombre):
    try:
        peliculas_df = cargar_csv('movies_dataset.csv')
        actores_df = cargar_csv("actores_dataset.csv")
        merged_df = pd.merge(actores_df, peliculas_df, on='id') # Realizar el join basado en el id de la película

        # Filtrar por el nombre del actor
        actor_peliculas = merged_df[merged_df['cast_name'].str.lower() == actor_nombre.lower()]

        if actor_peliculas.empty:
            return {'message': f"No se encontró información para el actor {actor_nombre}."}
        num_peliculas = actor_peliculas['id'].nunique() # Calcular la cantidad de películas
        suma_retorno = actor_peliculas['return'].sum() # Calcular la suma de los retornos
        promedio_retorno = actor_peliculas['return'].mean()# Calcular el promedio de los retornos
        return {
            'message': f"El actor {actor_nombre} ha participado en {num_peliculas} películas. El mismo ha conseguido un retorno de {suma_retorno} con un promedio de {promedio_retorno:.2f} por película."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def director_info(nombre_persona):
    try:
        # Cargar los DataFrames desde los archivos CSV
        peliculas_df = cargar_csv("movies_dataset.csv")
        produccion_df = cargar_csv("directores_dataset.csv")

        # Filtrar solo los registros donde el job es "Director"
        directores_df = produccion_df[produccion_df['crew_job'].str.lower() == 'director']
        directores_filtrados = directores_df[directores_df['crew_name'].str.lower() == nombre_persona.lower()]  # Filtrar por el nombre del director

        if directores_filtrados.empty:
            return {'message': f"No se encontró información para el director {nombre_persona}. Puede que la persona no sea un director o esté mal escrito."}

        merged_df = pd.merge(directores_filtrados, peliculas_df, on='id') # Realizar el join basado en movie_id

        # Calcular el éxito del director medido a través del retorno
        total_return = merged_df['return'].sum()
        average_return = merged_df['return'].mean()

        # Obtener la información detallada de las películas dirigidas
        peliculas_info = []
        for index, row in merged_df.iterrows():
            title = row['title']
            release_date = row['release_date']
            individual_return = row['return']
            budget = row['budget']
            revenue = row['revenue']
            peliculas_info.append({
                'title': title,
                'release_date': release_date,
                'individual_return': individual_return,
                'budget': budget,
                'revenue': revenue
            })

        return {
            'message': f"El director {nombre_persona} tiene un retorno total de {total_return} y un retorno promedio de {average_return:.2f}.las Películas que ha dirigido son:",
            'total_return': total_return,
            'average_return': average_return,
            'peliculas_info': peliculas_info,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Función para calcular similitudes combinadas bajo demanda
def calcular_combined_similarities_bajo_demanda(reference_movie, sampled_Modelo):
    try:
        sampled_Modelo= cargar_csv('sampled.csv')
        # Crear la columna 'combined_features' combinando 'genre', 'director' y 'actor'
        sampled_Modelo['combined_features'] = sampled_Modelo['genero'] + ' ' + sampled_Modelo['director'] + ' ' + sampled_Modelo['Actor_principal']
        print(sampled_Modelo['combined_features'].head())

        # Cálculo de similitudes textuales (TF-IDF)
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sampled_Modelo['combined_features'])

        # Cálculo de similitudes numéricas
        numeric_features = ['popularity', 'vote_average', 'vote_count', 'release_year']
        scaler = MinMaxScaler()
        scaled_numeric_features = scaler.fit_transform(sampled_Modelo[numeric_features])
        numeric_similarities = cosine_similarity(scaled_numeric_features)

        # Encontrar el índice de la película de referencia
        reference_index = sampled_Modelo.index[sampled_Modelo['title'].str.lower() == reference_movie.lower()].tolist()[0]

        # Calcular las similitudes combinadas
        text_similarities = cosine_similarity(tfidf_matrix[reference_index], tfidf_matrix).flatten()
        combined_similarities = (0.5 * text_similarities) + (0.5 * numeric_similarities[reference_index])

        return combined_similarities

    except Exception as e:
        raise Exception(f"No se pudieron calcular las similitudes combinadas bajo demanda: {str(e)}")
    
# Función para obtener recomendaciones de películas
def get_recommendations(reference_movie, Modelo, combined_similarities, top_n=5):
    try:
        Modelo = cargar_csv('Modelo.csv')
        reference_movie_lower = reference_movie.lower() # Convertir el título de la película de referencia a minúsculas
        Modelo['title_lower'] = Modelo['title'].str.lower()  # Convertir todos los títulos en el DataFrame a minúsculas

        # Encontrar el índice de la película de referencia
        reference_index = Modelo[Modelo['title_lower'] == reference_movie_lower].index[0]

        # Obtener las similitudes combinadas para la película de referencia
        similarities = combined_similarities[reference_index]

        # Obtener los índices de las películas más similares ordenadas
        similar_movies_indices = similarities.argsort()[::-1][1:(top_n + 1)]  # Excluye la película de referencia

        # Obtener los títulos de las películas recomendadas
        recommended_movies = Modelo.iloc[similar_movies_indices]['title'].tolist()

        # Eliminar la columna temporal 'title_lower' del DataFrame Modelo
        Modelo.drop(columns=['title_lower'], inplace=True)

        return {
            'message': f"Películas recomendadas para '{reference_movie}':",
            'recommended_movies': recommended_movies
        }

    except Exception as e:
        raise Exception(f"Error al obtener recomendaciones: {str(e)}")

@app.get("/peliculas_por_dia/")
def peliculas_por_dia_endpoint(nombre_dia: str):
    try:
        resultado = peliculas_estrenadas_por_dia(nombre_dia)
        return resultado
    except HTTPException as e:
        return {'error': e.detail}   
    
@app.get("/peliculas_por_mes/")
def peliculas_por_mes_endpoint(nombre_mes: str):
    try:
        resultado = peliculas_estrenadas_por_mes(nombre_mes)
        return resultado
    except HTTPException as e:
        return {'error': e.detail}    
    
@app.get("/informacion_pelicula/")
def informacion_pelicula_endpoint(nombre_pelicula: str):
    try:
        resultado = informacion_pelicula(nombre_pelicula)
        return resultado
    
    except HTTPException as e:
        return {'error': e.detail}
    
@app.get("/informacion_pelicula_votaciones/")   
def informacion_pelicula_votaciones_endpoint(nombre_pelicula: str):
    try:
        resultado = informacion_pelicula_votaciones(nombre_pelicula)
        return resultado
    
    except HTTPException as e:
        return {'error': e.detail}

@app.get("/informacion_actores_retorno/")
def informacion_actores_retorno_endpoint(actor_nombre: str):
    try:
        resultado = informacion_actores_retorno(actor_nombre)
        return resultado
    
    except HTTPException as e:
        return {'error': e.detail}
    

@app.get("/director_info/")
def director_info_endpoint(nombre_persona: str):
    try:
        resultado = director_info(nombre_persona)
        return resultado
    
    except HTTPException as e:
        return {'error': e.detail}
    

@app.post("/calculate_similarities/")
def calculate_similarities(movie_name: str, modelo_path: str):
    try:
        combined_similarities = calcular_combined_similarities_bajo_demanda(movie_name, modelo_path)
        return {"combined_similarities": combined_similarities.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations/")
def get_recommendations_api(movie_name: str, modelo_path: str, combined_similarities: list):
    try:
        recommendations = get_recommendations(movie_name, modelo_path, combined_similarities)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))