import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import json

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Determina o diretório onde está o executável ou o script
diretorio_app = os.path.dirname(sys.executable if hasattr(sys, 'frozen') else __file__)

# Obtém a chave de API do ambiente
chave_api = os.getenv('OPENAI_API_KEY')

# Cria um cliente OpenAI usando a chave de API fornecida
cliente = OpenAI(api_key=chave_api)

# Define o modelo do OpenAI a ser utilizado
modelo = 'gpt-4-1106-preview'

minhas_ferramentas = [
    {"type": "file_search"} # Ferramenta de busca de arquivos
    # A ferramenta file_search permite que o assistente pesquise por informações em arquivos que você fez upload para o ambiente de execução do assistente.
    # A ferramenta de busca de arquivos (file_search) é usada para permitir que o assistente faça consultas inteligentes em arquivos armazenados, com base em conteúdo semântico e não apenas palavras-chave exatas.
    # Quando um assistente está configurado para usar essa ferramenta, ele pode procurar e recuperar documentos ou partes de documentos que são semanticamente relevantes para uma consulta, usando a representação vetorial desses arquivos.
    # Quando um usuário faz uma pergunta ou pede informações, o assistente usa a ferramenta file_search para realizar uma busca dentro do armazenamento vetorial.
    # A entrada do usuário é transformada em um vetor (usando um modelo da OpenAI ou outro modelo de embeddings).
    # Esse vetor é comparado com os vetores armazenados para identificar os documentos ou trechos mais relevantes.
    # O modelo recupera informações relevantes de um conjunto de dados e gera uma resposta personalizada com base nesses dados, em vez de depender apenas de informações pré-treinadas ou de uma base fixa de conhecimento.
    # A busca de arquivos amplia o assistente com conhecimento externo ao seu modelo, como informações proprietárias de produtos ou documentos fornecidos pelos seus usuários.
    # A OpenAI automaticamente analisa e divide seus documentos, cria e armazena as representações (embeddings) e utiliza buscas baseadas em vetores e palavras-chave para recuperar conteúdo relevante e responder às consultas dos usuários.
]

def criar_armazenamento_vetorial():
    # Cria um novo armazenamento vetorial.
    # Esse repositório armazenará os vetores dos arquivos que o assistente vai precisar.
    armazenamento_vetorial = cliente.beta.vector_stores.create(name='armazenamento_vetorial_assistente_faq')
    # Os Vector Stores são bancos de dados especializados que armazenam os pedaços de texto (chunks) de arquivos de forma vetorial, permitindo buscas rápidas e precisas.
    # Cada vector store pode conter até 10.000 arquivos e pode ser conectado a um assistente ou a uma thread para fornecer resultados de pesquisa durante uma conversa.

    # Caminho para o diretório 'documentos' no projeto
    diretorio_documentos = os.path.join(diretorio_app, 'documentos')

    # Filtra os arquivos suportados (por exemplo, .txt, .json, .pdf)
    extensoes_validas = ['.txt', '.json', '.pdf']
    # Arquivos suportados: https://platform.openai.com/docs/assistants/tools/file-search/supported-files#supported-files

    # Cria uma lista com os caminhos dos arquivos contidos no diretório 'documentos'
    # Lista de caminhos dos arquivos que contêm os dados que queremos anexar ao armazenamento vetorial.
    caminho_arquivos = [
        os.path.join(diretorio_documentos, arquivo)
        for arquivo in os.listdir(diretorio_documentos)
        if os.path.isfile(os.path.join(diretorio_documentos, arquivo)) and
        any(arquivo.lower().endswith(extensao_valida) for extensao_valida in extensoes_validas) # Verificação de extensão
        # A função any() em Python recebe um iterável (como uma lista, tupla ou gerador) e retorna True se
        # qualquer um dos elementos desse iterável for verdadeiro (ou equivalente a True), e False caso contrário.
    ]

    # Abre cada arquivo no modo de leitura binária ('rb') e armazena os fluxos de arquivos (file streams) em uma lista.
    # Essa lista `fluxos_arquivos` é usada para carregar os arquivos para o armazenamento vetorial.
    fluxos_arquivos = [open(caminho, 'rb') for caminho in caminho_arquivos]

    # Faz o upload dos arquivos no armazenamento vetorial recém-criado.
    # O método `upload_and_poll` envia os arquivos para o armazenamento e espera a confirmação de upload.
    # O parâmetro `vector_store_id` usa o ID do armazenamento vetorial que criamos anteriormente.
    cliente.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=armazenamento_vetorial.id,
        files=fluxos_arquivos
    )

    # Retorna o armazenamento vetorial criado
    return armazenamento_vetorial

# Função para carregar as instruções
def carregar_instrucoes():
    """Carrega as instruções da assistente"""

    # Compoe o caminho do arquivo 'instrucoes.txt' dentro da pasta 'configuracoes' no diretório do aplicativo
    caminho_arquivo = os.path.join(diretorio_app, 'configuracoes', 'instrucoes.txt')
    try:
        # Tenta abrir o arquivo no caminho especificado
        with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
            # Lê o conteúdo do arquivo e armazena na variável 'instrucoes'
            instrucoes = arquivo.read()
        # Retorna o conteúdo do arquivo como resultado da função
        return instrucoes
    except FileNotFoundError:
        # Caso o arquivo não seja encontrado, exibe uma mensagem de erro especificando o caminho do arquivo
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        # Retorna None para indicar que o carregamento falhou
        return None
    except Exception as e:
        # Captura qualquer outra exceção que possa ocorrer ao tentar abrir ou ler o arquivo
        print(f"Erro ao carregar as instruções: {e}")
        # Retorna None para indicar que ocorreu um erro inesperado no processo
        return None

def criar_thread(armazenamento_vetorial):
    # O Thread representa uma conversa entre o assistente e o usuário.
    # Cada conversa será associada a um Thread, que armazena todas as mensagens trocadas entre ambos.
    return cliente.beta.threads.create(
        tool_resources={
            'file_search': {
                'vector_store_ids': [armazenamento_vetorial.id]
                # É importante incluir o parâmetro vector_store_ids para que o modelo saiba de onde buscar as informações relevantes.
                # Toda vez que você inicia uma thread (uma sessão de conversa) usando o Assistente, é importante vincular os documentos.
                # Isso é feito informando o vector_store_id na criação da thread, garantindo que as respostas do Assistente nesta sessão considerem a base de conhecimento definida.
            }
        }
    )

# Função para criar o assistente com as instruções carregadas
def criar_assistente(armazenamento_vetorial):
    """Cria um novo assistente com base nas instruções carregadas."""

    # Define as instruções que serão usadas pelo assistente (carregadas do arquivo de instruções)
    instrucoes = carregar_instrucoes()

    # Verifica se as instruções foram carregadas corretamente
    if instrucoes is None:
        print("Erro: Não foi possível carregar as instruções do assistente.")
        return None

    # Cria uma instância do assistente no OpenAI usando as instruções carregadas
    assistente = cliente.beta.assistants.create(
        # O serviço/recurso de Assistente da OpenAI e sua API de Assistente são soluções projetadas para
        # ajudar desenvolvedores a criar assistentes de IA personalizados que podem ser integrados a suas próprias aplicações.
        # Esses assistentes utilizam modelos de IA da OpenAI, como o GPT, e podem ser configurados para desempenhar tarefas específicas, com suporte a ferramentas e funcionalidades avançadas.
        name='assistente_faq', # Define o nome do assistente
        instructions=instrucoes, # Instruções carregadas do arquivo para o assistente
        # Essas instruções guiam o comportamento do assistente, por exemplo, especificando seu papel ou personalidade.
        model=modelo, # Especifica o modelo de linguagem a ser utilizado pelo assistente
        tools=minhas_ferramentas,
        # As ferramentas que podem ser anexadas ao assistente na Assistants API da OpenAI são funcionalidades adicionais
        # que permitem que o assistente execute tarefas específicas ou interaja com diferentes tipos de dados e sistemas.
        tool_resources={
            'file_search': {
                'vector_store_ids': [armazenamento_vetorial.id]
            }
            # Esta ferramenta permite que o assistente acesse, busque e processe arquivos.
            # Ela é parte de uma abordagem de RAG (Retrieval-Augmented Generation), que permite ao assistente pesquisar e acessar
            # informações em arquivos armazenados para ajudar a responder perguntas ou realizar tarefas.
            # Pode ser útil para consultar documentos grandes ou realizar buscas específicas dentro de conjuntos de arquivos.
            # Cada vector store pode conter até 10.000 arquivos e pode ser conectado a um assistente ou a uma thread para fornecer resultados de pesquisa durante uma conversa.
        }
        # O parâmetro tool_resources é um dicionário onde você define as ferramentas e recursos adicionais
        # que o assistente terá acesso para usar durante a interação com o usuário.
        # Neste caso, a chave 'file_search' indica que o assistente terá acesso à ferramenta de pesquisa de arquivos.
    )

    # Retorna o objeto do assistente criado para uso
    return assistente

# Serve para gerenciar a persistência dos dados relacionados ao assistente e a thread,
# evitando a necessidade de criar um novo assistente a cada execução do script.
def pegar_configuracoes():
    # Define o nome do arquivo JSON que será utilizado.
    nome_arquivo = os.path.join(diretorio_app, 'configuracoes', 'configuracoes.json')

    # Verifica se o arquivo 'configuracoes.json' não existe.
    if not os.path.exists(nome_arquivo):
        # Se o arquivo não existir, cria um novo armazenamento vetorial para armazenar os dados.
        armazenamento_vetorial = criar_armazenamento_vetorial()

        # Cria uma nova thread, que pode ser usada para gerenciar a conversa ou a interação com o assistente.
        thread = criar_thread(armazenamento_vetorial)

        # Cria um novo assistente que utilizará o armazenamento vetorial recém-criado para buscar informações.
        assistente = criar_assistente(armazenamento_vetorial)

        # Cria um dicionário para armazenar os dados que serão salvos no arquivo JSON.
        dados = {
            'id_assistente': assistente.id, # ID do assistente criado.
            'id_armazenamento_vetorial': armazenamento_vetorial.id,
            'id_thread': thread.id # ID da thread criada.
        }

        # Abre o arquivo 'configuracoes.json' em modo de escrita ('w') com codificação UTF-8.
        with open(nome_arquivo, 'w', encoding='utf-8') as arquivo:
            # Salva os dados no arquivo JSON, garantindo que caracteres não ASCII sejam corretamente representados.
            json.dump(dados, arquivo, ensure_ascii=False, indent=4)

        # Informa ao usuário que o arquivo foi criado com sucesso.
        print("Arquivo 'configuracoes.json' criado com sucesso.")

    # Tenta abrir e ler o arquivo 'configuracoes.json'.
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
            # Carrega os dados do arquivo JSON e os retorna como um dicionário.
            dados = json.load(arquivo)
            return dados
    # Captura a exceção caso o arquivo não seja encontrado.
    except FileNotFoundError:
        print("Arquivo 'configuracoes.json' não encontrado.")

# Chama a função pegar_configuracoes() para recuperar os dados do assistente e da thread.
assistente = pegar_configuracoes()

# Acessa o ID da thread armazenado no dicionário 'assistente' e o atribui à variável 'thread_id'.
# Isso permite que o código faça referência à thread já existente.
id_thread = assistente['id_thread']

# Acessa o ID do assistente armazenado no dicionário 'assistente' e o atribui à variável 'assistente_id'.
# Isso permite que o código utilize o assistente já existente.
id_assistente = assistente['id_assistente']

# Acessa a lista de IDs de arquivos armazenados no dicionário 'assistente' e a atribui à variável 'file_ids'.
# Isso permite que o código utilize os arquivos previamente carregados associados ao assistente.
id_armazenamento_vetorial = assistente['id_armazenamento_vetorial']

# Cria o loop de interação com o chatbot
def chatbot():
    """Função que interage com o modelo OpenAI para obter respostas baseadas no prompt do usuário."""

    # Define o número máximo de tentativas em caso de falha
    maximo_tentativas = 1
    # Contador de tentativas
    repeticao = 0

    print("\nBem-vindo ao SAC do Banco! Digite sua pergunta abaixo:")

    # Loop para tentar a comunicação com o modelo até obter sucesso ou atingir o máximo de tentativas
    while True:
        prompt = input('\nVocê: ')
        if prompt.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando o atendimento. Até mais!")
            break

        # Envia o prompt do usuário para a thread do assistente
        cliente.beta.threads.messages.create(
            thread_id=id_thread, # Identificador exclusivo da thread de conversa, utilizado para manter o contexto das interações
            role='user',
            content=prompt
        )

        # Executa o processamento do prompt pelo assistente
        run = cliente.beta.threads.runs.create(
            thread_id=id_thread, # Identificador exclusivo da thread de conversa, que associa a execução ao contexto da interação atual
            assistant_id=id_assistente # Identificador do assistente que irá processar o prompt, permitindo que o sistema saiba qual assistente está ativo na thread
        )

        # Como essa resposta não aparece de forma instantânea, temos que garantir que só vamos computar essa resposta
        # depois que o assistente responder aos questionamentos presentes na thread.
        # Verifica se o processamento do prompt foi concluído
        while run.status != 'completed':
            run = cliente.beta.threads.runs.retrieve(
                thread_id=id_thread, # Identificador da thread associada à execução atual, utilizado para garantir que o status verificado corresponde à conversa correta
                run_id=run.id # Identificador da execução específica que está sendo monitorada, permitindo que o sistema recupere o estado atual da execução
            )

        # Recupera a lista de mensagens da thread especificada e converte os dados em uma lista
        historico = list(cliente.beta.threads.messages.list(thread_id=id_thread).data)

        # Armazena a primeira mensagem do histórico, que geralmente é a resposta do assistente
        resposta = historico[0]

        # Retorna a resposta obtida do assistente
        print(f"Assistente: {resposta.content[0].text.value}")

chatbot()