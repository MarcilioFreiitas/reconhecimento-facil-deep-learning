

# Reconhecimento Fácil Deep Learning 🎥🤖

> Sistema unificado para **reconhecimento facial** com GPU opcional, **detecção corporal YOLOv8**, **rastreamento DeepSORT** e **detecção de quedas**, totalmente em Python, pronto para rodar em CPU ou GPU.

---

## 📋 Sumário

* [Sobre](#sobre)
* [✨ Funcionalidades](#-funcionalidades)
* [🚀 Tecnologias e Dependências](#-tecnologias-e-dependências)
* [⚙️ Instalação](#️-instalação)
* [🗂️ Estrutura do Projeto](#️-estrutura-do-projeto)
* [📝 Preparo do Banco Facial (Enroll)](#️-preparo-do-banco-facial-enroll)
* [🎯 Uso do Sistema Unificado](#-uso-do-sistema-unificado)
* [🔬 Suporte a GPU](#-suporte-a-gpu)
* [⚙️ Configurações Avançadas](#️-configurações-avançadas)
* [🤝 Contribuições](#-contribuições)
* [📄 Licença](#-licença)

---

## Sobre

Este repositório reúne dois módulos principais:

1. **enroll\_faces.py** – Extrai e agrupa embeddings faciais a partir de um *dataset* de imagens, usando InsightFace, com augmentation automático.
2. **unified\_recognizer.py** – Sistema principal: captura vídeo, faz detecção e rastreamento de corpos (YOLOv8 + DeepSORT), reconhecimento facial (InsightFace), e detecta quedas de forma robusta e performática. Indica no console se cada etapa está rodando em CPU ou GPU.

Ideal para aplicações de monitoramento, teleassistência, saúde, robótica e experimentação em visão computacional.

---

## ✨ Funcionalidades

* 💁‍♂️ **Reconhecimento Facial** com [InsightFace](https://github.com/deepinsight/insightface), pronto para GPU
* 🏃‍♂️ **Detecção Corporal** em tempo real com [YOLOv8](https://github.com/ultralytics/ultralytics) (PyTorch, CPU/GPU automático)
* 🔄 **Rastreamento DeepSORT** para corpos
* ⚠️ **Detecção de Quedas** heurística, adaptável
* 🖼️ **Data Augmentation** no enroll automático
* 📺 Suporte a vídeo local e webcam
* 🖥️ **Prints automáticos informando se o processamento está em CPU ou GPU**

---

## 🚀 Tecnologias e Dependências

* Python ≥ 3.8
* OpenCV
* NumPy
* [InsightFace](https://github.com/deepinsight/insightface) (CPU ou GPU)
* [Ultralytics YOLOv8](https://pypi.org/project/ultralytics/) (CPU ou GPU)
* [DeepSORT realtime](https://github.com/levan92/deep_sort_realtime)
* pickle
* mxnet (para InsightFace, use versão correta para seu CUDA/CPU)
* torch (PyTorch com ou sem CUDA)

**Instalação básica:**

```bash
pip install opencv-python numpy insightface ultralytics deep_sort_realtime
```

> Para usar GPU, veja a seção [🔬 Suporte a GPU](#-suporte-a-gpu).

---

## ⚙️ Instalação

```bash
# Clone este repositório
git clone https://github.com/MarcilioFreiitas/reconhecimento-facil-deep-learning.git
cd reconhecimento-facil-deep-learning

# (Recomendado) Crie e ative um ambiente virtual ou conda
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
# Ou, se não existir o arquivo, instale manualmente conforme acima
```

---

## 🗂️ Estrutura do Projeto

```
├── dataset/                    # Imagens brutas para enrollment
│   ├── pessoa1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── pessoa2.jpg
├── enroll_faces.py            # Script de preparo e enroll
├── unified_recognizer.py      # Sistema principal de detecção/rastreamento/quedas
├── trainer/
│   └── face_db.pickle         # Banco de embeddings gerado
└── README.md
```

---

## 📝 Preparo do Banco Facial (Enroll)

1. Organize suas imagens em `dataset/`, separadas por pessoa.
2. Execute:

```bash
python enroll_faces.py
```

* Informe o caminho do dataset e onde salvar o banco facial (opcional).
* O script gera variações, extrai embeddings e salva a média por pessoa.

---

## 🎯 Uso do Sistema Unificado

```bash
python unified_recognizer.py
```

* Informe o caminho do vídeo ou `0` para webcam.
* O sistema mostra no console, a cada execução, se o YOLOv8 (PyTorch) e o InsightFace (MXNet) estão usando **CPU ou GPU**.
* Janela com detecção de corpos, faces e alerta "QUEDA!" no vídeo.
* Pressione **Esc** para encerrar.

---

## 🔬 Suporte a GPU

**YOLOv8 e InsightFace suportam GPU, mas dependem do seu ambiente.**

### Como garantir o uso de GPU:

1. **Drivers NVIDIA** instalados e CUDA disponível.
2. **PyTorch** instalado com CUDA.

   * Veja seu CUDA com `nvidia-smi`.
   * Exemplo para CUDA 12.1:

     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
3. **MXNet** instalado na versão CUDA correta:

   * Para CUDA 12.1:

     ```bash
     pip install mxnet-cu121
     ```
   * Para CPU apenas:

     ```bash
     pip install mxnet
     ```
4. Rode seu script normalmente! O console informará se GPU está ativa.

---

## ⚙️ Configurações Avançadas

* Ajuste thresholds e parâmetros de queda/editando `unified_recognizer.py`.
* Troque o modelo YOLO alterando:

  ```python
  self.yolo_model = YOLO("yolov8n.pt")
  ```
* Veja mensagens de status no terminal: o script mostra se está usando CPU ou GPU em cada etapa.

---

## 🤝 Contribuições

Contribuições são bem‐vindas!

1. Fork este repositório
2. Crie uma branch feature: `git checkout -b feature/nome-da-feature`
3. Commit suas mudanças: `git commit -m "✨ Nova feature"`
4. Push: `git push origin feature/nome-da-feature`
5. Abra um Pull Request


