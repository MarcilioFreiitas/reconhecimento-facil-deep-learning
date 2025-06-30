# Reconhecimento Fácil Deep Learning 🎥🤖

> Sistema unificado para **reconhecimento facial**, **detecção corporal**, **rastreamento assíncrono** e **detecção de quedas**, tudo em Python.

---

## 📋 Sumário

- [Sobre](#sobre)  
- [✨ Funcionalidades](#-funcionalidades)  
- [🚀 Tecnologias e Dependências](#-tecnologias-e-dependências)  
- [⚙️ Instalação](#️-instalação)  
- [🗂️ Estrutura do Projeto](#️-estrutura-do-projeto)  
- [📝 Preparando o Dataset e Enrollamento](#️-preparando-o-dataset-e-enrollamento)  
- [🎯 Uso do Reconhecedor](#-uso-do-reconhecedor)  
- [⚙️ Configurações Adicionais](#️-configurações-adicionais)  
- [🤝 Contribuições](#-contribuições)  
- [📄 Licença](#-licença)  

---

## Sobre

Este repositório reúne dois módulos:

1. **enroll_faces.py** – Extrai e agrupa embeddings faciais a partir de um _dataset_ de imagens, usando InsightFace + data augmentation.  
2. **unified_recognizer.py** – Captura vídeo (arquivo ou webcam), detecta rostos, corpos, rastreia ambos em threads separadas e dispara alertas de queda com heurística leve.

Ideal para aplicações de vigilância, monitoramento de idosos ou experiências interativas que demandem percepção visual em tempo real.

---

## ✨ Funcionalidades

- 💁‍♂️ **Reconhecimento Facial** com [InsightFace](https://github.com/deepinsight/insightface)  
- 🏃‍♂️ **Detecção Corporal** em tempo real via [YOLOv8](https://github.com/ultralytics/ultralytics)  
- 🔄 **Rastreamento Assíncrono** multi‐thread (face & corpo)  
- ⚠️ **Detecção de Quedas** por análise de histórico de bounding‐boxes  
- 🖼️ **Augmentation Automático** no processo de enroll (flip, brilho, rotação)  
- 📺 Suporte a vídeo local e webcam  

---

## 🚀 Tecnologias e Dependências

- Python ≥ 3.8  
- OpenCV  
- NumPy  
- [InsightFace](https://github.com/deepinsight/insightface)  
- [Ultralytics YOLOv8](https://pypi.org/project/ultralytics/)  
- concurrent.futures (thread pool)  
- pickle (serialização de embeddings)  

Instale via `pip`:

```bash
pip install opencv-python numpy insightface ultralytics
```

---

## ⚙️ Instalação

```bash
# Clone este repositório
git clone git@github.com:MarcilioFreiitas/reconhecimento-facil-deep-learning.git
cd reconhecimento-facil-deep-learning

# Crie e ative um ambiente virtual (opcional, mas recomendado)
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Instale dependências
pip install -r requirements.txt
```

> Se não houver `requirements.txt`, instale manualmente conforme [Dependências](#-tecnologias-e-dependências).

---

## 🗂️ Estrutura do Projeto

```
├── dataset/                    # Imagens brutas para enrollment
│   ├── pessoa1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── pessoa2.jpg             # ou diretamente na raiz
├── enroll_faces.py            # Script de preparo e enroll
├── unified_recognizer.py      # Sistema principal de detecção/rastreamento
├── trainer/
│   └── face_db.pickle         # Banco de embeddings gerado
└── README.md
```

---

## 📝 Preparando o Dataset e Enrollamento

1. Organize suas imagens em `dataset/`.  
   - Subpastas: cada pasta é um **rótulo** (nome da pessoa).  
   - Arquivos isolados: nome antes do “.” vira rótulo.  
2. Execute o enroll:

```bash
python enroll_faces.py
```

- Informe o caminho do `dataset` ou pressione Enter (padrão: `dataset`).  
- Informe onde salvar o banco de dados (padrão: `trainer/face_db.pickle`).  

O script gerará variações de cada imagem (flip, brilho, rotação), extrairá embeddings e salvará a média por pessoa.

---

## 🎯 Uso do Reconhecedor

```bash
python unified_recognizer.py
```

- **Entrada**: caminho para vídeo ou `0` para webcam.  
- **Flags de configuração** (opcionais, editar no código):
  - `min_similarity`: limiar de confiança para reconhecimento  
  - `face_detection_interval`: intervalo de detecção de faces  
  - `scale_factor`: escala de redimensionamento do frame  
- **Saída**: janela interativa com:
  - Caixas e rótulos de rostos  
  - Caixas de corpos  
  - Alerta “**QUEDA!**” abaixo do corpo em caso de evento detectado  

Pressione **Esc** para encerrar.

---

## ⚙️ Configurações Adicionais

- Caso utilize GPU e tenha problemas com _duplicate libs_, já configuramos:
  ```python
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  ```
- Ajuste thresholds de queda em `unified_recognizer.py`:
  ```python
  self.height_drop_threshold = 0.3        # % da altura inicial
  self.aspect_ratio_threshold = 0.6       # h/w menor = deitado
  self.fall_consecutive_frames = 3        # persistência (frames)
  ```
- Para usar outro modelo YOLO, altere na inicialização:
  ```python
  self.yolo_model = YOLO("yolov8n.pt")
  ```

---

## 🤝 Contribuições

Contribuições são muito bem‐vindas!  
1. Fork este repositório  
2. Crie uma branch feature: `git checkout -b feature/nome-da-feature`  
3. Commit suas mudanças: `git commit -m "✨ Nova feature"`  
4. Push na branch: `git push origin feature/nome-da-feature`  
5. Abra um Pull Request  

---

## 📄 Licença

Este projeto está sob a [MIT License](LICENSE).  

---

> Desenvolvido por **Marcilio Freiitas** 🚀  
> Dúvidas ou feedback? Abra uma [issue](https://github.com/MarcilioFreiitas/reconhecimento-facil-deep-learning/issues).
