#  Calcular Temperatura - Noel Reyes21-2021


# Lista de días
Dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# Pedimos la cantidad de días a registrar
cantidad_dias = int(input("¿Cuántos días vas a registrar? (Lunes-Domingo): "))

# Inicializamos contadores y acumuladores 
suma_temperaturas = 0
dias_frios = 0
moderados = 0
calidos = 0

# Usamos un slice [0:cantidad_dias] para recorrer solo los días pedidos
for i in range(cantidad_dias):
    dia_actual = Dias[i]
    temp = float(input(f"Ingrese la temperatura del {dia_actual}: "))
    suma_temperaturas += temp

    # Clasificación de Temperatura
    if temp < 15:
        print(f"El {dia_actual} fue un día Frío.")
        dias_frios += 1
    elif 15 <= temp <= 25:
        print(f"El {dia_actual} fue un día moderado.")
        moderados += 1
    else:
        print(f"El {dia_actual} fue un día cálido.")
        calidos += 1

# Cálculo del promedio
promedio = suma_temperaturas / cantidad_dias

print("\n--- Resumen de la Semana ---")
print(f"- Días Fríos: {dias_frios}")
print(f"- Días Moderados: {moderados}")
print(f"- Días Cálidos: {calidos}")
print(f"- Temperatura Promedio: {promedio:}°C")








#   Encuestadora Ciencias de Datos- Noel Reyes21-2021

# Pedimos la cantidad de personas  a registrar
cantidad_personas = int(input("¿Cuántas personas vas a registrar?: "))

#Inicializamos  contadores y acumuladores 
suma_calificaciones = 0
insatisfechos = 0
edad_minima = 18 
nombre_joven = ""

for i in range(cantidad_personas):
    print(f"\n--- Encuestado #{i+1} ---")
    nombre = input("Nombre: ")
    edad = int(input("Edad: "))
    calificacion = int(input("Calificación (1-5): "))

    # 1. Acumular calificación para el promedio
    suma_calificaciones += calificacion

    # 2. Contar insatisfechos (menor a 3)
    if calificacion < 3:
        print("Gracias por tu honestidad, trabajaremos en mejorar.")
        insatisfechos += 1
    elif calificacion == 3:
        print("Gracias por tu opinión.")
    else:
        print("¡Nos alegra saber que estás satisfecho!")

    # 3. Lógica para encontrar al más joven
    if edad < edad_minima:
        edad_minima = edad
        nombre_joven = nombre

# Cálculos finales
if cantidad_personas > 0:
    promedio = suma_calificaciones / cantidad_personas
    
    print("\n" + "="*30)
    print("RESUMEN DE LA ENCUESTA")
    print("="*30)
    print(f"1. Promedio de calificaciones: {promedio:.2f}")
    print(f"2. Personas insatisfechas: {insatisfechos}")
    print(f"3. La persona más joven es: {nombre_joven} ({edad_minima} años)")
else:
    print("\n"+ "No se registraron mas personas.")