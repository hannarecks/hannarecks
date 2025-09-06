import json
import os
import glob

import google.generativeai as genai
from dotenv import load_dotenv


class ExamAnalyzer:
    def __init__(self):
        # Carrega variáveis de ambiente
        load_dotenv()

        # Configura API do Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY não encontrada! Crie um arquivo .env com sua chave da API.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def get_analysis_prompt(self, exam_content: str) -> str:
        """
        Retorna o prompt para análise do exame pelo Gemini.
        """
        return f"""Você é um especialista em análise de relatórios de mamografia. Sua tarefa é ler o relatório fornecido e extrair as informações solicitadas na estrutura abaixo. Siga ATENTAMENTE as "Diretrizes de Interpretação".

Estrutura de Saída:
Cisto:
- Presente ou Ausente
- Localização e/ou tamanho do cisto

Nódulo:
- Presente ou Ausente
- Localização e/ou tamanho do nódulo

Calcificação:
- Presente ou Ausente
- Localização e/ou tamanho da calcificação

Microcalcificação:
- Presente ou Ausente
- Localização e/ou tamanho da microcalcificação

BI-RADS: [valor]

Outras citações a avaliar: [observações adicionais relevantes]

Diretrizes de Interpretação:

1.  Identificação Geral de Achados:
    * Para cada categoria principal (Cisto, Nódulo, Calcificação, Microcalcificação), determine o Status (Presente ou Ausente) e, se presente, extraia a Localização e o Tamanho.
    * Se informações específicas não estiverem disponíveis no texto, utilize "[sem referência no texto]".

2.  Diferenciação e Reclassificação Nódulo/Cisto:
    * Definições Básicas: Nódulos são estruturas predominantemente sólidas; cistos são estruturas predominantemente líquidas.
    * Reclassificação de Nódulo Mamográfico para Cisto Ecográfico:
        * Condição de Aplicabilidade: Esta sub-regra de reclassificação aplica-se exclusivamente quando o relatório atual indica que achados de uma MAMOGRAFIA foram subsequentemente (ou conjuntamente) avaliados por ECOGRAFIA (Ultrassonografia) e esta ecografia está esclarecendo a natureza de um achado mamográfico. A simples menção de ambos os exames no histórico não ativa esta regra se não houver uma reclassificação explícita de um achado específico.
        * Ação de Reclassificação: Se, e somente se, a condição acima for atendida, e um achado inicialmente descrito como "nódulo" na mamografia for claramente identificado e reclassificado pela ecografia como "cisto" (ex: "cisto simples", "achado mamográfico corresponde a cisto ao ultrassom", "natureza cística confirmada pela ecografia"), então, para essa lesão específica:
            * Cisto: Status (Presente), com os detalhes fornecidos (idealmente da ecografia).
            * Nódulo: Status (Ausente).
        * Quando NÃO há Reclassificação (Nódulo permanece Nódulo, Cisto permanece Cisto):
            * Se a ecografia confirmar um achado mamográfico como um nódulo sólido (ex: "nódulo sólido correspondente ao achado mamográfico").
            * Se o relatório for apenas de mamografia (sem ecografia complementar descrita para o achado) ou apenas de ecografia (sem referência a um achado mamográfico sendo reclassificado).
            * Se o achado for descrito como um complexo sólido-cístico (ver abaixo).
    * Complexos Sólido-Císticos: Se uma lesão for descrita como tendo componentes tanto sólidos quanto císticos (ex: "nódulo complexo", "cisto com componente sólido", "lesão sólido-cística"), ela deve ser reportada como PRESENTE para AMBAS as categorias: Cisto E Nódulo, com as respectivas descrições e tamanhos, se disponíveis.
    * Nódulos e Cistos como Achados Distintos e Múltiplos: Se o relatório descrever um nódulo e um cisto como duas (ou mais) lesões separadas e distintas (não uma reclassificação de uma única lesão), ambos devem ser extraídos individualmente com status "Presente" e seus respectivos detalhes.
    * Detalhamento: Sempre descreva o tipo do cisto ou/e nódulo, caso presente no relatório.


3.  Múltiplos Achados do Mesmo Tipo:
    * Quando houver múltiplos cistos ou múltiplos nódulos, reporte TODOS, priorizando: a) Achados classificados como suspeitos pelo relatório, b) Achados de maior tamanho, c) Achados com características atípicas mencionadas. Liste suas localizações e tamanhos.

4.  Diferenciação entre Calcificações e Microcalcificações:
    * Calcificações: estruturas maiores, geralmente descritas como "grosseiras", "distróficas", "vasculares".
    * Microcalcificações: estruturas menores, frequentemente descritas como "puntiformes", "pleomórficas", "lineares", "agrupadas", "em cluster".
    * Se o relatório mencionar "microcalcificações" especificamente, classifique como microcalcificações. Se mencionar apenas "calcificações" (e a descrição não sugerir microcalcificações), classifique como calcificações.

IMPORTANTE: Retorne a resposta APENAS em formato JSON válido, seguindo exatamente esta estrutura:

{{
    "cisto": {{
        "presente": true/false,
        "detalhes": "localização e/ou tamanho"
    }},
    "nodulo": {{
        "presente": true/false,
        "detalhes": "localização e/ou tamanho"
    }},
    "calcificacao": {{
        "presente": true/false,
        "detalhes": "localização e/ou tamanho"
    }},
    "microcalcificacao": {{
        "presente": true/false,
        "detalhes": "localização e/ou tamanho"
    }},
    "bi_rads": "valor",
    "outras_citacoes": "observações adicionais relevantes"
}}

RELATÓRIO DO EXAME:
{{exam_content}}
"""

    def analyze_exam_with_gemini(self, exam_content: str) -> dict:
        """
        Analisa o conteúdo do exame usando Google Gemini.
        """
        prompt = self.get_analysis_prompt(exam_content)
        response = self.model.generate_content(prompt)

        # Extrai JSON da resposta
        response_text = response.text.strip()

        # Remove possíveis marcadores de código se existirem
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]

        # Parse do JSON
        analysis_data = json.loads(response_text.strip())
        return analysis_data

    def process_text_file(self, text_file_path: str) -> dict:
        """
        Processa um arquivo de texto e analisa seu conteúdo.
        """
        # Verifica se o arquivo existe
        if not os.path.exists(text_file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {text_file_path}")

        # Lê o conteúdo do arquivo de texto
        with open(text_file_path, 'r', encoding='utf-8') as f:
            exam_content = f.read()

        # Analisa com Gemini
        analysis = self.analyze_exam_with_gemini(exam_content)

        # Salva o resultado em JSON
        base_name = os.path.splitext(text_file_path)[0]
        json_file = f"{base_name}_analise.json"

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        print(f"Análise salva em: {json_file}")
        return analysis

    def process_directory(self, directory_path: str) -> None:
        """
        Processa todos os arquivos de texto em um diretório.
        """
        # Verifica se o diretório existe
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")
        
        # Encontra todos os arquivos .txt no diretório
        text_files = glob.glob(os.path.join(directory_path, "*.txt"))
        
        if not text_files:
            print(f"Nenhum arquivo .txt encontrado no diretório: {directory_path}")
            return
        
        print(f"Encontrados {len(text_files)} arquivos .txt para processar")
        
        # Processa cada arquivo
        for file_path in text_files:
            try:
                print(f"\nProcessando: {os.path.basename(file_path)}")
                self.process_text_file(file_path)
            except Exception as e:
                print(f"ERRO ao processar {file_path}: {e}")


def main():
    print("=== Analisador de Exames Médicos ===\n")

    try:
        # Inicializa o analisador
        print("Inicializando analisador...")
        analyzer = ExamAnalyzer()

        # Define o diretório de exames
        exames_dir = "exames"
        
        # Verifica se o diretório existe, cria se não existir
        if not os.path.exists(exames_dir):
            os.makedirs(exames_dir)
            print(f"Diretório '{exames_dir}' criado.")
            print(f"Por favor, adicione os arquivos .txt dos exames neste diretório e execute o programa novamente.")
            return

        # Processa os arquivos do diretório
        analyzer.process_directory(exames_dir)
        
        print("\nProcessamento concluído!")

    except Exception as e:
        print(f"\nERRO: {e}")


if __name__ == "__main__":
    main()
