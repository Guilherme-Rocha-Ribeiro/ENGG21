import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import 

# Carregar os dados
url = 'https://docs.google.com/spreadsheets/d/12e_aMg3fiuOQCBbw8eknkWN6Qbo-5rXu8WECJ4f0g-k/edit?gid=0#gid=0'
url = url.replace('/edit?','/export?format=csv&')
dados = pd.read_csv(url, decimal=',')
dados.head()

pass
# Funções auxiliares
def predict_outputs(X, theta):
    """Make predictions using linear model: ŷ = Xθ"""
    return np.dot(X, theta)

def calculate_residual(X_sample, theta, y_actual):
    """Calculate residual (error) for a single data point: y_hat - y_actual"""
    return np.dot(X_sample, theta) - y_actual

def least_squares(X, y):
    """Normal equation: θ = (XᵀX)⁻¹Xᵀy"""
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

def gradient_descent(X, y, initial_theta, learning_rate, n_epochs):
    """
    Perform gradient descent to optimize linear regression parameters.
    """
    theta_current = initial_theta.copy()
    errors = np.zeros((n_epochs, 1))
    
    for epoch in range(n_epochs):
        total_squared_error = 0.0

        for i in range(len(X)):
            # Calculate error for current data point
            residual = calculate_residual(X[[i], :], theta_current, y[i])
            
            # Update parameters: θ = θ - α * 2 * e * Xᵀ
            gradient = 2 * residual * X[[i], :].T
            theta_current = theta_current - (learning_rate * gradient)
            
            total_squared_error += residual**2

        errors[epoch] = total_squared_error
    
    return theta_current, errors

def neuronio(u, params):
    """Neurônio classificador com função degrau"""
    # Nucleo: Combinação linear
    x = u.dot(params)
    # Função de ativação: transformação não linear (degrau)
    y = np.where(x > 0, 1, 0)
    return y.flatten()

# Preparar os dados de entrada
def preparar_dados(dados):
    """Preparar matriz de características com bias"""
    # Converter para numpy array
    U = dados.values  # Ignorar primeira coluna (número)
    
    # Adicionar coluna de bias (1's)
    UI = np.hstack((U, np.ones((U.shape[0], 1))))
    
    return UI

# Definir diferentes classes de saída
def definir_classes(tipo, numeros):
    """
    Definir vetor de saída Y para diferentes classificações
    """
    if tipo == "pares":
        return np.array([1 if n % 2 == 0 else 0 for n in numeros])
    elif tipo == "multiplos3":
        return np.array([1 if n % 3 == 0 else 0 for n in numeros])
    elif tipo == "primos":
        primos = [2, 3, 5, 7]  # Números primos de 1 dígito
        return np.array([1 if n in primos else 0 for n in numeros])
    elif tipo == "fibonacci":
        fibonacci = [0, 1, 2, 3, 5, 8]  # Números de Fibonacci de 1 dígito
        return np.array([1 if n in fibonacci else 0 for n in numeros])
    elif tipo == "maiores5":
        return np.array([1 if n > 5 else 0 for n in numeros])
    else:
        raise ValueError("Tipo de classificação não reconhecido")

# Função para treinar e avaliar o modelo
def treinar_e_avaliar(tipo_classificacao, dados):
    """Treinar e avaliar o neurônio para uma classificação específica"""
    
    # Preparar dados
    numeros = np.arange(100)
    UI = preparar_dados(dados)
    Y = definir_classes(tipo_classificacao, numeros)
    
    print(f"\n=== Classificação: {tipo_classificacao.upper()} ===")
    print(f"Números: {numeros}")
    print(f"Classes: {Y}")
    print(f"0 = Não pertence, 1 = Pertence")
    
    # Inicializar parâmetros
    theta_0 = np.random.randn(UI.shape[1], 1) * 0.1
    theta_0[-1] = 0  # bias inicial
    
    # Predição inicial
    y_pred_inicial = neuronio(UI, theta_0)
    acuracia_inicial = np.mean(y_pred_inicial == Y) * 100
    print(f"\nAcurácia inicial: {acuracia_inicial:.1f}%")
    
    # Treinar com gradient descent
    theta_otimizado, erros = gradient_descent(
        X=UI, 
        y=Y, 
        initial_theta=theta_0, 
        learning_rate=1e-2, 
        n_epochs=400
    )
    
    # Predição final
    y_pred_final = neuronio(UI, theta_otimizado)
    acuracia_final = np.mean(y_pred_final == Y) * 100
    
    print(f"Acurácia final: {acuracia_final:.1f}%")
    print(f"Parâmetros finais: {theta_otimizado.flatten()}")
    
    # Mostrar resultados detalhados
    print("\nResultados detalhados:")
    for i, num in enumerate(numeros):
        status = "✓" if y_pred_final[i] == Y[i] else "✗"
        print(f"Número {num}: Real={Y[i]}, Predito={y_pred_final[i]} {status}")
    
    return theta_otimizado, erros, acuracia_final, y_pred_final

# Executar para diferentes classificações
classificacoes = ["pares", "multiplos3", "primos", "fibonacci", "maiores5"]

resultados = {}

plt.figure(figsize=(15, 10))

for idx, classificacao in enumerate(classificacoes):
    theta, erros, acuracia, y_pred = treinar_e_avaliar(classificacao, dados)
    
    resultados[classificacao] = {
        'theta': theta,
        'acuracia': acuracia,
        'y_pred': y_pred
    }
    
    # Plotar curva de erro
    plt.subplot(2, 3, idx + 1)
    plt.plot(erros, '.-')
    plt.title(f'Erro - {classificacao.capitalize()}\nAcurácia: {acuracia:.1f}%')
    plt.xlabel('Época')
    plt.ylabel('Erro Quadrático')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Análise comparativa
print("\n" + "="*50)
print("ANÁLISE COMPARATIVA DOS RESULTADOS")
print("="*50)

for classificacao in classificacoes:
    acc = resultados[classificacao]['acuracia']
    print(f"{classificacao.upper():<12}: {acc:>5.1f}% de acurácia")

# Testar com novos dados (simulação)
def testar_novo_numero(theta, numero_array, tipo):
    """Testar o classificador com um novo padrão"""
    # Adicionar bias
    padrao_com_bias = np.hstack((numero_array, [1]))
    predicao = neuronio(padrao_com_bias.reshape(1, -1), theta)
    return predicao[0]

# Demonstrar capacidade de generalização
print("\n" + "="*50)
print("TESTE DE GENERALIZAÇÃO")
print("="*50)

# Usar o classificador de números pares como exemplo
theta_pares = resultados['pares']['theta']
print("\nTestando classificador de números pares:")

# Criar alguns padrões de teste simples
padroes_teste = [
    ([1,1,1,1,0,1,1,0,1,1,0,1,1,1,1], 0),  # Padrão similar a 0 (par)
    ([0,1,0,0,1,0,0,1,0,0,1,0,0,1,0], 1),  # Padrão similar a 1 (ímpar)
]

for padrao, esperado in padroes_teste:
    predicao = testar_novo_numero(theta_pares, np.array(padrao), "pares")
    status = "✓" if predicao == esperado else "✗"
    print(f"Padrão → Esperado: {esperado}, Predito: {predicao} {status}")

# Discussão do modelo
print("\n" + "="*50)
print("DISCUSSÃO DO MODELO")
print("="*50)
print("""
CARACTERÍSTICAS DO MODELO:
- Perceptron simples com função de ativação degrau
- Treinamento via Descida do Gradiente
- Capaz de aprender fronteiras de decisão lineares

LIMITAÇÕES:
- Só pode aprender problemas linearmente separáveis
- Sensível à inicialização dos parâmetros
- Pode convergir para mínimos locais

OBSERVAÇÕES:
- O desempenho varia conforme a complexidade da classificação
- Classes mais simples (pares/ímpares) tendem a ter melhor performance
- A convergência depende da taxa de aprendizado e número de épocas
""")