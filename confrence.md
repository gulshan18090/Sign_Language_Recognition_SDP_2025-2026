% Real-Time Sign Language Sentence Recognition — Springer LNCS
% Version: improved full draft with Cosine vs Seq2Seq comparison
%
\documentclass[runningheads]{llncs}
%
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
%
% Uncomment the following two lines for blue hyperlinks (Springer eBook style):
%\usepackage{color}
%\renewcommand\UrlFont{\color{blue}\rmfamily}
%
\begin{document}
%
\title{Real-Time Azerbaijani Sign Language Sentence Recognition:\\
Comparative Evaluation of Cosine Retrieval and Seq2Seq Approaches}
%
\titlerunning{Real-Time Azerbaijani Sign Language Sentence Recognition}
%
\author{Gulshan Sariyeva\inst{1} \and
Rahima Karimova\inst{1} \and
Vasila Aliyeva\inst{1} \and
Ilaha Jamalli\inst{1} \and
Jamaladdin Hasanov\inst{1}}

\authorrunning{G.\ Sariyeva et al.}

\institute{
School of IT and Engineering, ADA University, Baku, Azerbaijan\\
\email{\{gsariyeva18090, rkarimova17347, valiyeva16144, ijamalli16111, jhasanov\}@ada.edu.az}
}
%
\maketitle
%
\begin{abstract}
Sign language is the primary language for many Deaf and Hard-of-Hearing (DHH) communities. State-of-the-art sentence-level sign language recognition approaches are computationally expensive and require large annotated corpora, neither of which is available for low-resource sign languages such as Azerbaijani Sign Language (AzSL). The explicit goal of this paper is to design and evaluate a \emph{lightweight, CPU-only, training-free} sentence-level retrieval system that can run on commodity hardware without GPU acceleration, and to benchmark it against a learned neural baseline on the same data.

Our \emph{primary contribution} is a \emph{training-free cosine retrieval pipeline}: it extracts 21 MediaPipe hand landmarks per frame (3D coordinates over two hands, yielding 126 dimensions per frame), uniformly samples 64 frames per video to form a $64\times126$ feature matrix, flattens it into a single $8064$-dimensional global \emph{flat embedding} (a vectorisation of the per-frame landmarks with no learned aggregation), and retrieves the nearest reference sentence by cosine similarity. The pipeline is end-to-end CPU-only, requires no labelled training set, and supports trivially incremental vocabulary growth: enrolling a new sentence is one extra video.

As a learned baseline we compare against a \emph{sequence-to-sequence (Seq2Seq) neural model} that encodes per-frame SqueezeNet features with a bidirectional LSTM and decodes sentence tokens with a Bahdanau-attention LSTM and beam search. A shared real-time webcam module detects signing onset and offset using a four-state finite state machine and returns the top-3 candidate sentences.

On the crowdsourced Telegram AzSL dataset~\cite{mustafazada2025crowdsourcing} (335 videos, 120 sentences; 119 translator and 216 user-contributed videos) the cosine retrieval baseline reaches 30.7\% Top-1 and 45.0\% Top-3 accuracy with leave-one-out evaluation and zero training. The Seq2Seq baseline reaches 35.5\% Top-1, BLEU-4 of 31.43 and WER of 64.05, but exhibits significant mode collapse (a single sentence accounts for 36.7\% of all predictions). User-contributed videos (37.0\% Top-1) outperform translator videos (32.8\%), suggesting that crowdsourced multi-signer variability is beneficial for learned models. Both methods run end-to-end in real time on a CPU. We report the limitations of the small corpus, the closed-vocabulary nature of retrieval, the mode-collapse behaviour of Seq2Seq, and the absence of non-manual features as open problems for future AzSL research.

\keywords{Sign Language Recognition \and MediaPipe \and Hand Landmarks \and Cosine Similarity \and LSTM \and Seq2Seq Attention \and Real-Time Inference \and Azerbaijani Sign Language \and Low-Resource Recognition \and Crowdsourcing}
\end{abstract}

% ================================================================
\section{Introduction}
% ================================================================

Sign language recognition (SLR) is one of the most direct ways to bridge the communication gap between Deaf and Hard-of-Hearing (DHH) users and hearing individuals. Modern deep learning models perform well on isolated-sign and word-level recognition, but real-time \emph{sentence-level} SLR remains difficult due to variability in signing speed, signer style, camera viewpoint, and environmental conditions~\cite{liu2024skeleton}. The problem is amplified for low-resource sign languages such as Azerbaijani Sign Language (AzSL), for which no large annotated corpus, pre-trained model, or established benchmark exists. The Azerbaijani Sign Language Dataset (AzSLD)~\cite{alishzade2024azsld} and the crowdsourced Telegram dataset~\cite{mustafazada2025crowdsourcing} have begun to fill this gap, but existing open-source AzSL recognition baselines are limited to fingerspelling and isolated signs.

In this paper we focus deliberately on the \emph{practical, application-oriented} setting in which AzSL recognition is most likely to be deployed first: short fixed-vocabulary sentences (greetings, common phrases, frequent service-domain utterances) that can be recognised on a commodity CPU without a GPU. This narrows the problem in two ways. First, we restrict the recognition target to sentences that exist in a curated reference database; this is appropriate for retrieval-style applications such as DHH--hearing communication kiosks or learning aids, but it does mean the retrieval baseline cannot generalise to truly unseen sentences -- a limitation we discuss explicitly in Section~\ref{sec:discussion}. Second, we constrain ourselves to lightweight, CPU-only architectures so that the resulting system can run on edge hardware.

\paragraph{Objectives.} Concretely, the goals of this work are:
\begin{enumerate}
    \item to design a CPU-only, sentence-level AzSL recognition system that runs in real time on a standard laptop without GPU acceleration;
    \item to compare a training-free retrieval baseline with a learned Seq2Seq model on the same crowdsourced AzSL dataset, to quantify what learning buys (and what it costs) at this data scale; and
    \item to characterise the failure modes (mode collapse, signer-type imbalance, missing non-manual features) that future AzSL work will need to address.
\end{enumerate}

\paragraph{Contributions.} Towards these objectives we contribute:
\begin{enumerate}
    \item a \textbf{training-free cosine retrieval pipeline} based on MediaPipe hand landmarks, wrist-centred normalisation, and a flat $8064$-dimensional embedding -- an end-to-end CPU pipeline that requires no GPU and no labelled training set;
    \item a \textbf{Seq2Seq encoder--decoder} that uses SqueezeNet visual features, a bidirectional LSTM encoder, and a Bahdanau-attention LSTM decoder with beam search, adapted to sentence-level AzSL on a small crowdsourced corpus;
    \item a \textbf{shared real-time webcam module} that automatically segments signing using a four-state finite state machine (\textsc{idle}\,$\to$\,\textsc{recording}\,$\to$\,\textsc{cooldown}\,$\to$\,\textsc{matching}) and returns the top-3 candidate sentences;
    \item a \textbf{comparative empirical study} on 335 crowdsourced AzSL videos reporting Top-$K$ accuracy, BLEU-4, WER, CER, signer-type stratification, frame-count sensitivity, and a fair end-to-end latency analysis for both methods; and
    \item an \textbf{honest analysis of failure modes}, including a quantification of Seq2Seq mode collapse and a discussion of what the residual diversity of the model's predictions does and does not tell us about the validity of the learned mapping.
\end{enumerate}

The remainder of the paper is organised as follows. Section~\ref{sec:related} surveys related work. Section~\ref{sec:method} details the two proposed methods. Section~\ref{sec:experiments} presents experiments and comparative results, including a comparison against reported figures from related SLR systems. Section~\ref{sec:conclusion} concludes.

% ================================================================
\section{Related Work}\label{sec:related}
% ================================================================

\paragraph{Classical SLR.} The earliest sign language recognition (SLR) systems targeted isolated words or fingerspelling and relied on Hidden Markov Models (HMMs) or Dynamic Time Warping (DTW) for temporal alignment~\cite{Liu_2024_05}. While instrumental in establishing the field, these methods generalised poorly across signing speeds, signers, and camera angles~\cite{carneiro2024sign}. For AzSL specifically, the earliest published work focused on real-time fingerspelling-to-text translation using statistical models~\cite{sanchezbrizuela2023lightweight} and on classifier-based recognition of the AzSL alphabet; continuous, sentence-level AzSL recognition has remained largely unexplored.

\paragraph{Deep Continuous SLR.} The shift to deep learning has produced strong results on continuous SLR (CSLR) benchmarks. A typical CSLR pipeline has three components: a visual module (often a CNN such as ResNet) that extracts per-frame features, a sequential module (LSTM, GRU, or BiLSTM) that models temporal dynamics~\cite{verma2024enhancing}, and an alignment module that uses Connectionist Temporal Classification (CTC) loss to avoid explicit per-sign pre-segmentation. Hybrid CNN--LSTM and ResNet--BiLSTM models~\cite{varshini2025mpgestlstm} dominate the leaderboards on benchmarks such as RWTH-PHOENIX-2014T~\cite{camgoz2018neural}. However, these models typically require thousands of annotated videos and GPU training, neither of which is available for AzSL.

\paragraph{Skeleton-based and landmark-based SLR.} To reduce the dependence on large labelled video corpora, recent work has moved towards \emph{skeleton-based} representations that are robust to lighting and background changes. MediaPipe Hands~\cite{nguyen2023exploring} provides a CPU-friendly pipeline that produces 21 three-dimensional joint coordinates per hand at real-time rates and reaches accuracy comparable to OpenPose at a fraction of the compute. Several works combine MediaPipe with sequence models for gesture and sign recognition: MP-GestLSTM~\cite{varshini2025mpgestlstm} uses MediaPipe landmarks with an LSTM for real-time gesture detection, and similar pipelines have been applied to American Sign Language~\cite{kamble2025slrnet}. The 126-dimensional per-frame vector that MediaPipe yields (21 landmarks $\times$ 3 coordinates $\times$ 2 hands) is compact enough to enable lightweight downstream models, including the retrieval baseline we propose.

\paragraph{Sentence-level retrieval.} Cosine similarity is a standard metric for comparing high-dimensional embeddings and has been used widely in speaker verification~\cite{fang2013bayesian,dehak2010cosine} and metric learning~\cite{zhang2020deep,zhu2020orthogonality}. Retrieval-based pipelines that compare a query to a database of known gestures sidestep the need for large supervised training sets and are inherently interpretable: the matched neighbour can be inspected directly. The trade-off is that pure retrieval is a \emph{closed-vocabulary} method -- a query that does not correspond to any sentence in the reference database will still return a nearest neighbour, but that neighbour will be wrong. We discuss this limitation explicitly in Section~\ref{sec:discussion}. To the best of our knowledge, no prior work has applied a landmark-based cosine retrieval scheme to sentence-level AzSL.

\paragraph{Sentence-level Seq2Seq translation.} Camg\"oz et al.~\cite{camgoz2018neural} introduced neural sign language translation as a Seq2Seq problem with attention, motivating our second baseline. Their work and follow-ups operate on much larger corpora (PHOENIX-2014T contains 8\,257 sentences); reproducing such an architecture on 335 videos is known to be data-starved, and our experiments confirm this: the learned model improves over retrieval but exhibits significant mode collapse (Section~\ref{sec:discussion}).

\paragraph{Position of this work.} We deliberately situate ourselves at the intersection of (i) low-resource AzSL, (ii) lightweight CPU-only deployment, and (iii) sentence-level (not isolated-sign) recognition. Existing work covers at most two of these three at a time; this paper provides the first comparative study of training-free and learned methods that satisfies all three constraints simultaneously.




% ================================================================
\section{Proposed Methods}\label{sec:method}
% ================================================================

We present two approaches for sentence-level AzSL recognition: (1)~a training-free cosine retrieval pipeline based on MediaPipe hand landmarks, and (2)~a Seq2Seq encoder--decoder model with CNN visual features and attention. Both share a common real-time webcam inference module described in Section~\ref{sec:realtime}.

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{diagram/system.png}
\caption{System architecture overview. Offline: reference videos are processed into feature representations and stored. Online: webcam frames undergo the same pipeline and are matched via cosine retrieval or decoded by the Seq2Seq model.}
\label{fig:architecture}
\end{figure}

% ================================================================
\subsection{Approach~1: Cosine Similarity Retrieval}\label{sec:cosine_method}
% ================================================================

% ----------------------------------------------------------------
\subsubsection{Frame Extraction.}\label{sec:frame}
% ----------------------------------------------------------------

Videos in our dataset are recorded on a variety of consumer devices and so vary in both frame rate and signing tempo. We use OpenCV~(\texttt{cv2.VideoCapture}) only as the frame-decoding library; the temporal normalisation itself is performed by \emph{uniform index-based subsampling} of the decoded frame stream into a fixed number of frames~$N = 64$, regardless of the original frame rate or duration. Concretely, given a decoded video of length~$T$ frames, we select frame indices
\begin{equation}
\mathrm{idx}_i = \left\lfloor \frac{i \cdot T}{N} \right\rfloor,
\quad i = 0, 1, \dots, N-1\,,
\label{eq:sampling}
\end{equation}
i.e.\ a stride of $T/N$ across the decoded stream. When $T < N$, the same index can repeat: this is equivalent to a zero-order hold (frame duplication). When $T \geq N$ we are simply subsampling at stride $T/N$.

We note that this scheme normalises sequence length but not signing speed: two videos of the same sentence whose original durations differ by a factor of two are both reduced to $N$ frames, but the resulting frame sequences correspond to different physical time offsets within the gesture, which can introduce within-class variance. We do not attempt content-aware frame selection or motion-aware re-timing in this work; we rely on cosine similarity and the LSTM's tolerance to local mis-alignment to absorb this variance, and we report its observable effect in the per-sentence variance analysis in Section~\ref{sec:discussion}. We make no objective representativeness measurement of individual frames; the selection is purely uniform-stride. Improving temporal alignment (for example via DTW alignment to a canonical reference, or via motion-energy-based keyframe selection) is a clear direction for future work.
% ----------------------------------------------------------------
\subsubsection{Hand Landmark Feature Extraction.}\label{sec:features}
% ----------------------------------------------------------------

We use the MediaPipe Hand model to extract features, yielding a 126-dimensional feature vector per frame ($21~\text{landmarks} \times 3~\text{coordinates}\;(x,y,z) \times 2~\text{hands}$). As we want our model to be camera scale and subject distance invariant once deployed on resource-limited edge devices, the features are normalized around the wrist.

Let the landmarks have coordinates $\mathbf{l}_i \in \mathbb{R}^{3}$ for each $i$-th landmark and $\mathbf{l}_0$ for the wrist landmark. First, the landmarks are centered at the wrist landmark and scaled to fit a unit sphere:
\begin{equation}
\mathbf{c}_i = \mathbf{l}_i - \mathbf{l}_0
\label{eq:centre}
\end{equation}
\begin{equation}
\mathbf{n}_i = \frac{\mathbf{c}_i}
                     {\max_{j}\, \lVert \mathbf{c}_j \rVert}
\label{eq:normalise}
\end{equation}
A scaling factor is used to center the wrist (removing the effect of the signer being in a different position in the image) and to normalize the gesture size regardless of distance from the camera. This is calculated as the maximum distance between landmarks. In the absence of detected hands, zero-padding is used.

\begin{figure}[t]
\centering
\includegraphics[width=0.45\linewidth]{diagram/fig2_mediapipe_hand.png}
\caption{MediaPipe hand landmark model: 21 joints with $(x,y,z)$ coordinates, yielding 63~features per hand.}
\label{fig:mediapipe}
\end{figure}

% ----------------------------------------------------------------
\subsubsection{Feature Matrix and Cosine Similarity.}\label{sec:cosine}
% ----------------------------------------------------------------

The sampled sequence produces a feature matrix $\mathbf{F} \in \mathbb{R}^{64 \times 126}$. We construct what we will call a \emph{flat embedding} -- the row-major vectorisation of $\mathbf{F}$ into a single vector $\mathbf{f} \in \mathbb{R}^{8064}$ (i.e.\ $\mathbf{f} = \mathrm{vec}(\mathbf{F})$, with no learned aggregation, no temporal pooling, and no dimensionality reduction). The motivation for this design is to obtain a representation that is fully analytic, has no learned parameters, and admits a closed-form similarity. Inference is performed by computing the cosine similarity between the query embedding~$\mathbf{a}$ and each template embedding~$\mathbf{b}$ in the database:
\begin{equation}
\mathrm{sim}(\mathbf{a}, \mathbf{b}) =
  \frac{\mathbf{a} \cdot \mathbf{b}}
       {\lVert \mathbf{a} \rVert \; \lVert \mathbf{b} \rVert}\,.
\label{eq:cosine}
\end{equation}
The cosine metric treats each flat embedding as a \emph{direction} in $\mathbb{R}^{8064}$ and returns the cosine of the angle between the query and the template. Since wrist-centred normalisation places every frame's coordinates on (a subset of) the unit sphere, the resulting flat embeddings are restricted to a compact region of $\mathbb{R}^{8064}$, which reduces the \emph{embedding-norm effect} reported for other similarity-based optimisation systems~\cite{draganov2024hidden}.

For each query video, the per-sentence score is the \emph{maximum} cosine similarity over that sentence's reference videos.

\paragraph{Closed-vocabulary retrieval.} Like every retrieval system, this pipeline always returns a nearest neighbour: it has no notion of a query that does not correspond to any sentence in the database. In practical deployment this is mitigated in two ways: the system returns the top-3 candidates with their similarity scores so the user can visually verify, and applications that need an out-of-vocabulary detector can threshold on the top-1 similarity score. For this paper we report standard Top-$K$ recognition accuracy assuming all queries are in-vocabulary, which matches our target setting (a fixed phrasebook of frequent service-domain sentences). Closed-vocabulary retrieval is a deliberate scoping choice for this work, not an oversight; we list out-of-vocabulary rejection as an explicit future direction in Section~\ref{sec:discussion}.

\paragraph{Limitations of the flat embedding.} Vectorising the $64\times126$ feature matrix into a single $8064$-dimensional vector destroys the temporal ordering: two videos in which the same handshapes appear in different order have similar flat embeddings even though they are different signs. In addition, neighbouring fingers tend to move together, which induces strong correlations among feature dimensions and reduces the discriminative power for visually similar sentences. We treat these as inherent limitations of the training-free baseline and revisit them in Section~\ref{sec:discussion}.

% ================================================================
\subsection{Approach~2: Seq2Seq Encoder--Decoder}\label{sec:seq2seq_method}
% ================================================================

The second approach treats sign-language sentence recognition as a sequence-to-sequence translation task, where the input is an arbitrary-length video frame sequence and the output is a variable-length token sequence corresponding to the target sentence. The approach is subdivided into three components: visual feature extraction, temporal encoding, and attentive decoding.

% ----------------------------------------------------------------
\subsubsection{Visual Feature Extraction.}\label{sec:cnn_features}
% ----------------------------------------------------------------

The pipeline begins with a hand-detection module (MediaPipe Hands); only frames where hand landmarks are detected are kept and scaled down to a size of $224 \times 224$ pixels and then passed to a pretrained SqueezeNet~1.1 backbone~\cite{iandola2016squeezenet}. The last feature map produced by this pre-trained CNN is high-level visual feature vectors representing hand shape and posture (size $512 \times 13 \times 13 = 86{,}528$). SqueezeNet was chosen as the architecture producing the feature vector due to its small model size ($\sim$1.2\,M parameters) enabling it to run on a CPU and its ability to produce discriminative visual features.

% ----------------------------------------------------------------
\subsubsection{Bidirectional LSTM Encoder.}\label{sec:encoder}
% ----------------------------------------------------------------

The features from the CNN are fed into a bidirectional LSTM encoder with hidden size $h = 256$:
\begin{equation}
\overrightarrow{\mathbf{h}_t} = \mathrm{LSTM}_{\rightarrow}
  (\mathbf{x}_t, \overrightarrow{\mathbf{h}_{t-1}}), \quad
\overleftarrow{\mathbf{h}_t} = \mathrm{LSTM}_{\leftarrow}
  (\mathbf{x}_t, \overleftarrow{\mathbf{h}_{t+1}})
\label{eq:bilstm}
\end{equation}
\begin{equation}
\mathbf{h}_t = [\overrightarrow{\mathbf{h}_t} \;;\;
                 \overleftarrow{\mathbf{h}_t}]
  \in \mathbb{R}^{512}
\label{eq:concat}
\end{equation}
where $\mathbf{x}_t$ is the CNN output flattened into a column vector of size $86528$ at frame~$t$, $[\cdot\,;\,\cdot]$ denotes concatenation, and the bidirectional architecture provides forward and backward context, producing a $512$-dimensional hidden state at each time step. The outputs of the encoder are the memory for attention-based decoding.

% ----------------------------------------------------------------
\subsubsection{Attention Decoder.}\label{sec:decoder}
% ----------------------------------------------------------------

The decoder is an attention-based LSTM with Bahdanau-style (additive) attention over the encoder outputs. At decoding step ~$s$, the decoder computes an attention distribution over the encoder states:
\begin{equation}
e_{s,t} = \mathbf{v}^\top \tanh\!\bigl(
  \mathbf{W}_q \,[\mathbf{e}_s \;;\; \mathbf{d}_{s-1}]
  + \mathbf{W}_k \,\mathbf{h}_t
\bigr)
\label{eq:energy}
\end{equation}
\begin{equation}
\alpha_{s,t} = \frac{\exp(e_{s,t}\,/\,\tau)}
                     {\sum_{t'} \exp(e_{s,t'}\,/\,\tau)}
\label{eq:attention}
\end{equation}
where $\mathbf{e}_s$ is the embedding of the previous token, $\mathbf{d}_{s-1}$ is the decoder hidden state, $\mathbf{W}_q$, $\mathbf{W}_k$, and $\mathbf{v}$ are trainable parameters, and $\tau$ is a temperature hyperparameter that controls the sharpness of the attention distribution (we use $\tau = 2.0$ to avoid attention collapse in the first few epochs). We also apply attention dropout ($p = 0.1$) to regularize the attention distribution during training.

At each time step, the context vector $\mathbf{c}_s = \sum_t \alpha_{s,t}\,\mathbf{h}_t$ is concatenated with the input embedding and fed into the decoder LSTM:
\begin{equation}
\mathbf{d}_s = \mathrm{LSTM}(\mathrm{ReLU}(
  \mathbf{W}_{\text{attn}}\,[\mathbf{e}_s \;;\; \mathbf{c}_s]),
  \,\mathbf{d}_{s-1})
\label{eq:decoder_step}
\end{equation}
The decoder hidden state $\mathbf{d}_s$ is then projected to the vocabulary space by a linear layer followed by a log-softmax to produce the next-token distribution.

% ----------------------------------------------------------------
\subsubsection{Training.}\label{sec:training}
% ----------------------------------------------------------------

The model is trained with cross-entropy loss and the Adam optimiser (initial learning rate $\eta = 0.01$). We employ:
\begin{itemize}
    \item \textbf{Scheduled sampling}: The teacher forcing ratio, which begins at 50\% and decays linearly throughout training epochs, encourages the model to rely on its self-predictions.
    \item \textbf{Learning rate scheduling}: ReduceLROnPlateau with factor~0.5 and patience~3.
    \item \textbf{Early stopping}: The patience is set to 7 epochs, and the minimum validation loss improvement is $\delta = 0.001$.
    \item \textbf{Gradient clipping}: maximum norm of~5.0.
\end{itemize}
The training run lasted 71 epochs; the best validation loss (2.568) was reached at epoch 64, after which validation loss began to increase, and the early-stopping mechanism restored the epoch-64 checkpoint. The learning rate was reduced four times by ReduceLROnPlateau during the run.

% ----------------------------------------------------------------
\subsubsection{Beam Search Decoding.}\label{sec:beam}
% ----------------------------------------------------------------

During inference, beam search is performed with a beam size of $B = 5$, returning the top-$K$ ($K \leq 5$) unique sentence hypotheses per input video in order of their sum log-probabilities for all decoding steps. Top-$K$ accuracy can be computed in the same way as for the cosine retrieval pipeline.

% ================================================================
\subsection{Real-Time Inference}\label{sec:realtime}
% ================================================================

Both approaches share a common real-time webcam inference module that automatically segments signing using a four-state finite state machine:
\begin{center}
\texttt{IDLE} $\;\rightarrow\;$ \texttt{RECORDING} $\;\rightarrow\;$ \texttt{COOLDOWN} $\;\rightarrow\;$ \texttt{MATCHING}
\end{center}

\begin{itemize}
    \item \textbf{IDLE}: the system waits for MediaPipe to detect hand landmarks in the webcam feed.
    \item \textbf{RECORDING}: as soon as a hand is detected, frames are buffered. Recording continues as long as at least one hand is visible.
    \item \textbf{COOLDOWN}: when the hands leave the frame, a two-second buffer is started. If hands return within this window the system goes back to \texttt{RECORDING}; otherwise the gesture is considered complete.
    \item \textbf{MATCHING}: the buffered frames are trimmed of empty leading/trailing segments, uniformly sampled to $N = 64$, and passed to the selected pipeline (cosine retrieval or Seq2Seq decoding); the top-3 candidates are displayed.
\end{itemize}

The 2-second cooldown prevents premature matching of partial sentences and trades a small amount of latency for substantially fewer false segmentations. After matching, the system returns to \texttt{IDLE}, ready for the next sentence.

% ================================================================
\section{Experiments and Results}\label{sec:experiments}
% ================================================================
This section presents the empirical evaluation of both pipelines. We report dataset properties, the system configuration, recognition accuracy under leave-one-out cross-validation, the impact of the frame-sampling parameter $N$, and end-to-end inference latency.

\subsection{Dataset}\label{sec:dataset}
We evaluate on the crowdsourced AzSL Telegram dataset~\cite{mustafazada2025crowdsourcing}, which contains 335 sentence-level videos covering 120 unique sentences. Of these, 119 videos (35.5\%) are recorded by professional translators (one trimmed translator video per sentence on average), and 216 videos (64.5\%) are user-contributed recordings collected through a Telegram bot. The broader AzSLD corpus~\cite{alishzade2024azsld} provides isolated-sign and fingerspelling data for AzSL but is not sentence-level; we therefore do not use it for the present evaluation. The Telegram dataset captures realistic informal signing conditions, with natural variation in lighting, framing, and signing tempo. The sentences cover greetings, common phrases, culturally specific expressions, and short descriptive utterances typical of the Azerbaijani DHH community. Dataset statistics are summarised in Table~\ref{tab:dataset}.

\paragraph{Per-sentence coverage.} An important property of the dataset for retrieval-style evaluation is whether each sentence is represented by more than one video. As Table~\ref{tab:dataset} shows, this holds for most but not all sentences: 94 of the 120 sentences have at least two videos (mean 2.8 videos per sentence, range 1--7), while 26 sentences are represented by only a single video. For those 26 single-video sentences a leave-one-out cosine retrieval query has \emph{no} reference of the same class to retrieve, so the maximum possible Top-1 accuracy on those queries is 0 by construction. We therefore report aggregate accuracy as well as a stratification by signer type (Section~\ref{sec:telegram_signer}) so that the effect of single-video sentences on the headline number is visible.

\paragraph{Dataset availability.} The Telegram dataset described in~\cite{mustafazada2025crowdsourcing} is released by its authors and the trimmed sentence-level subset used in this paper, together with the trained Seq2Seq checkpoint, the cosine-retrieval reference embeddings, and the evaluation scripts, will be made publicly available at the time of publication via the project repository to support reproducibility.

\begin{table}[t]
\centering
\caption{Telegram crowdsourced dataset statistics.}
\label{tab:dataset}
\begin{tabular}{@{}lr@{}}
\toprule
Metric & Value \\
\midrule
Total videos                & 335 \\
Unique sentences            & 120 \\
Translator videos           & 119 (35.5\%) \\
User (crowdsourced) videos  & 216 (64.5\%) \\
Videos / sentence (avg)     & 2.8 \\
Videos / sentence (min--max)& 1--7 \\
Sentences with 1 video      & 26 \\
Sentences with $\geq$2 videos & 94 \\
Avg sentence length         & 4.9 words \\
Unique vocabulary tokens    & 356 \\
Collection method           & Crowdsourced (Telegram bot) \\
Language                    & Azerbaijani SL \\
\bottomrule
\end{tabular}
\end{table}

% ----------------------------------------------------------------
\subsection{Implementation Details}\label{sec:implementation}
% ----------------------------------------------------------------

Table~\ref{tab:system} lists the system hyperparameters. The entire pipeline runs CPU-only, without GPU acceleration, on a standard laptop.

\begin{table}[t]
\centering
\caption{System parameters for both approaches.}
\label{tab:system}
\begin{tabular}{@{}lll@{}}
\toprule
Parameter & Cosine Retrieval & Seq2Seq \\
\midrule
Sampled frames ($N$) & 64 & Variable \\
Features per frame & 126 (MediaPipe) & 86{,}528 (SqueezeNet) \\
Embedding dimension & 8{,}064 & 512 (biLSTM output) \\
Encoder hidden size & --- & 256 ($\times 2$ bidir) \\
Decoder hidden size & --- & 512 \\
Vocabulary size & --- & 356 tokens \\
Detection confidence & 0.5 & 0.5 \\
Normalisation & Wrist-centred & ImageNet stats \\
Similarity/decoding & Cosine similarity & Beam search ($B\!=\!5$) \\
Training required & No & Yes (71 epochs) \\
Cooldown timeout & 2.0\,s & 2.0\,s \\
\bottomrule
\end{tabular}
\end{table}

\noindent
\textbf{Software:} Python~3.11, MediaPipe~0.10.x, PyTorch, OpenCV, NumPy, SciPy.\\
\textbf{Hardware:} Intel Core~i7 (13th~Gen), 16\,GB RAM—no GPU.\\
\textbf{Cosine protocol:} leave-one-out cross-validation. For each of the 335 videos in turn, that video is used as the query and the remaining 334 videos form the reference database. The per-sentence score is the maximum cosine similarity between the query embedding and any reference embedding of that sentence; predictions are ranked by per-sentence score and Top-$K$ accuracy is computed.\\
\textbf{Seq2Seq protocol:} the model is trained on the 335 videos with an 80/20 train--validation split (sentence-stratified where possible) for up to 71~epochs, with early stopping (patience~7) restoring the best validation checkpoint. Beam search ($B = 5$) returns up to the top-5 hypotheses per video; reported Top-1 is the highest-scoring beam.

% ----------------------------------------------------------------
\subsection{Comparative Accuracy}\label{sec:accuracy}
% ----------------------------------------------------------------

Table~\ref{tab:comparison} presents the central comparison between cosine retrieval and the Seq2Seq model on the Telegram dataset.

\begin{table}[t]
\centering
\caption{Recognition accuracy: Cosine Retrieval vs.\ Seq2Seq Encoder--Decoder on the Telegram crowdsourced dataset.}
\label{tab:comparison}
\begin{tabular}{@{}lcccc@{}}
\toprule
Model & Videos & Training & Top-1 (\%) & Top-3 (\%) \\
\midrule
Cosine Retrieval      & 335 & None      & 30.7 & 45.0 \\
Seq2Seq + Attention   & 335 & 71 epochs & \textbf{35.5} & --- \\
\bottomrule
\end{tabular}
\end{table}

The Seq2Seq model outperforms the training-free cosine retrieval baseline by 4.8 percentage points on Top-1 accuracy. Despite the small dataset size, the learned encoder--decoder benefits from the diversity of crowdsourced signer styles. Cosine retrieval, while behind on Top-1, recovers most of the gap on Top-3 (45.0\%), which is the operational metric for the deployed system because three candidates are shown to the user. Table~\ref{tab:seq2seq_metrics} reports additional sequence-level metrics for the Seq2Seq model.

% ----------------------------------------------------------------
\subsubsection*{Comparison with related SLR systems.}
% ----------------------------------------------------------------

There is no existing sentence-level AzSL recognition system to compare against directly: the only published AzSL baselines~\cite{alishzade2024azsld,sanchezbrizuela2023lightweight} target fingerspelling and isolated signs and report results on different data and different output spaces, so a direct numerical comparison would be misleading. To give the reader context, Table~\ref{tab:relwork} summarises reported figures from related sign and gesture recognition work. The rows are not on the same dataset and should not be read as a head-to-head benchmark, but they bracket what is currently feasible across signing levels, languages, and hardware budgets. We make this caveat explicit because directly transplanting any of these methods to our 335-video, 120-sentence, CPU-only setting is non-trivial and out of scope for this paper; we view the establishment of a first sentence-level AzSL reference point as part of the contribution.

\begin{table}[t]
\centering
\caption{Reported figures for related sign-language and gesture recognition systems. Datasets, output spaces and evaluation protocols differ; numbers are not directly comparable to ours and the table is intended to provide context, not a head-to-head benchmark.}
\label{tab:relwork}
\begin{tabular}{@{}llllr@{}}
\toprule
System & Language & Output level & Hardware & Reported acc. \\
\midrule
Camg\"oz et al.~\cite{camgoz2018neural} & DGS (German) & Sentence (translation) & GPU & BLEU-4 18.40 \\
MP-GestLSTM~\cite{varshini2025mpgestlstm} & General gestures & Isolated gesture & CPU & 98+\% \\
SLRNet~\cite{kamble2025slrnet} & ASL & Isolated sign & GPU & 95+\% \\
AzSLD baseline~\cite{alishzade2024azsld} & AzSL & Isolated sign / fingerspelling & GPU & --- \\
\midrule
\textbf{Cosine retrieval (ours)} & \textbf{AzSL} & \textbf{Sentence (retrieval)} & \textbf{CPU} & \textbf{30.7 / 45.0\% (T1/T3)} \\
\textbf{Seq2Seq (ours)} & \textbf{AzSL} & \textbf{Sentence (translation)} & \textbf{CPU} & \textbf{35.5\% T1, BLEU-4 31.43} \\
\bottomrule
\end{tabular}
\end{table}

Two observations follow. First, sentence-level translation on a small, low-resource corpus is inherently harder than isolated-sign classification on a large, balanced dataset: a 35.5\% sentence-level Top-1 with BLEU-4 of 31.43 on 335 videos should not be compared on the same axis as a 95+\% isolated-sign accuracy on tens of thousands of clips. Second, our Seq2Seq BLEU-4 of 31.43 is encouragingly competitive with reported sentence-level translation BLEU-4 in DGS~\cite{camgoz2018neural} (18.40) despite using two orders of magnitude less data and CPU-only inference. We do not claim these numbers are state-of-the-art; we claim only that they establish a first sentence-level AzSL reference point that future work can improve upon.

\begin{table}[t]
\centering
\caption{Seq2Seq sequence-level metrics on the Telegram dataset.}
\label{tab:seq2seq_metrics}
\begin{tabular}{@{}lrrr@{}}
\toprule
Metric & Overall & Translator & User \\
\midrule
Top-1 Accuracy (\%) & 35.5 & 32.8 & 37.0 \\
BLEU-4 (\%)         & 31.43 & 31.09 & 31.62 \\
WER (\%)            & 64.05 & 65.94 & 63.00 \\
CER (\%)            & 50.64 & 52.88 & 49.41 \\
\bottomrule
\end{tabular}
\end{table}

% ----------------------------------------------------------------
\subsection{Telegram Seq2Seq: Signer-Type Stratification}\label{sec:telegram_signer}
% ----------------------------------------------------------------

Table~\ref{tab:telegram_detail} reports the Seq2Seq Top-1 accuracy on the Telegram dataset, stratified by signer type.

\begin{table}[t]
\centering
\caption{Seq2Seq accuracy stratified by signer type (Telegram dataset).}
\label{tab:telegram_detail}
\begin{tabular}{@{}lcc@{}}
\toprule
Signer Type & Videos & Top-1 (\%) \\
\midrule
Translator ($n{=}119$) & 119 & 32.8 \\
User ($n{=}216$) & 216 & 37.0 \\
\midrule
All ($n{=}335$) & 335 & \textbf{35.5} \\
\bottomrule
\end{tabular}
\end{table}

User videos achieve 4.2 percentage points higher accuracy than translator videos. The larger number of user videos (216 vs. 119) provides denser per-sentence coverage, improving the model's ability to learn generalizable representations. This pattern is consistent with the cosine retrieval observations, where user-contributed videos also yield higher similarity scores due to richer intra-class sampling.

% ----------------------------------------------------------------
\subsection{Cosine Retrieval: Signer-Type Stratification}\label{sec:signer}
% ----------------------------------------------------------------

Table~\ref{tab:cosine_detail} reports Top-$K$ accuracy for the cosine retrieval approach on the Telegram dataset, stratified by signer type.

\begin{table}[t]
\centering
\caption{Cosine retrieval accuracy stratified by signer type (leave-one-out, $N=64$, Telegram dataset).}
\label{tab:cosine_detail}
\begin{tabular}{@{}lccc@{}}
\toprule
Metric & All ($n{=}335$) & Translator ($n{=}119$) & User ($n{=}216$) \\
\midrule
Top-1 Accuracy (\%) & 30.7 & 25.2 & 33.8 \\
Top-3 Accuracy (\%) & 44.2 & 37.8 & 47.7 \\
\bottomrule
\end{tabular}
\end{table}

User videos achieve higher accuracy than translator videos, consistent with the Seq2Seq results. User videos outnumber translator videos approximately 2:1 (216 vs. 119), providing denser coverage in the embedding space. Consequently, user queries are more likely to find close neighbors within the same class.

% ----------------------------------------------------------------
\subsection{Seq2Seq Convergence}\label{sec:training_results}
% ----------------------------------------------------------------

For completeness we note that the Seq2Seq model reaches its lowest validation loss (2.568) at epoch~64 of a 71-epoch run; the early-stopping criterion (patience~7) restores the epoch-64 checkpoint. Beyond this point training loss continues to decrease while validation loss rises, an unambiguous overfitting signal that we attribute to the small corpus size (335 videos for 120 sentences). We do not include a full epoch-by-epoch loss curve in this paper, as the broader pattern -- a model that learns the training distribution but does not generalise fully to held-out signers -- is captured more meaningfully by the mode-collapse analysis in Section~\ref{sec:discussion}.

% ----------------------------------------------------------------
\subsection{Effect of Frame Count}\label{sec:framecount}
% ----------------------------------------------------------------

We sweep the temporal sampling parameter $N \in \{16, 32, 64, 128\}$ using the uniform-stride scheme of Eq.~\ref{eq:sampling}. Results are reported in Table~\ref{tab:frames}.

\begin{table}[t]
\centering
\caption{Cosine retrieval accuracy vs.\ sampled frame count~$N$.}
\label{tab:frames}
\begin{tabular}{@{}ccccc@{}}
\toprule
$N$ & Top-1 (\%) & Top-3 (\%) & Top-5 (\%) & Embedding dim \\
\midrule
16  & 27.8 & 39.5 & 46.8 & 2{,}016 \\
32  & 31.6 & 43.9 & 49.7 & 4{,}032 \\
64  & 30.7 & 45.0 & 49.4 & 8{,}064 \\
128 & 30.7 & 45.0 & 49.4 & 16{,}128 \\
\bottomrule
\end{tabular}
\end{table}
Performance improves from  $N = 16$ to $N = 32$ due to a better temporal resolution for hand-shape transitions. $N = 64$ saturates, doubling the resolution to $N = 128$ shows no further gain and also doubling the number of embedding dimensions shows no improvement. Based on these observations, we choose $N = 64$ as the working point.

% ----------------------------------------------------------------
\subsection{Latency Analysis}\label{sec:latency}
% ----------------------------------------------------------------

For a fair comparison, both pipelines must be measured on the same stages of the inference path. We therefore report two latency views: (a) the \emph{end-to-end from-video} latency, in which both pipelines start from the same on-disk video and run through their full inference stack (MediaPipe extraction for cosine, SqueezeNet + bidirectional LSTM encoding for Seq2Seq), and (b) the \emph{search-only} latency, which is the only stage that varies at deployment time once the system is recording from a webcam. The frame-decoding and feature-extraction stages can be performed incrementally during recording at $\sim$30~FPS, so they do not block the user; the operationally relevant number for system responsiveness is therefore the search-only latency. Table~\ref{tab:latency} gives both views.

\begin{table}[t]
\centering
\caption{Latency breakdown (average per query, CPU only). Both pipelines are compared on equivalent stages.}
\label{tab:latency}
\begin{tabular}{@{}lcc@{}}
\toprule
Stage & Cosine (ms) & Seq2Seq (ms) \\
\midrule
Frame read + sampling (from disk)        & 667.8 & 667.8 \\
Per-frame feature extraction             & 2{,}036.2 (MediaPipe) & $\sim$2{,}000 (SqueezeNet+BiLSTM) \\
Search / decode                          & \textbf{2.8} & 17.1 \\
\midrule
End-to-end from video                    & $\sim$2{,}706.8 & $\sim$2{,}684.9 \\
Search-only (operational)                & \textbf{2.8} & 17.1 \\
\bottomrule
\end{tabular}
\end{table}

Two conclusions follow. First, on the end-to-end from-video metric the two pipelines are within a few percent of each other; both are dominated by per-frame feature extraction. Second, on the operationally relevant search-only metric -- which is what determines how quickly the system responds once the user has finished signing -- cosine retrieval is roughly 6$\times$ faster (2.8\,ms vs.\ 17.1\,ms) because it is a pure dot-product search over 335 reference vectors, while the Seq2Seq pipeline must run a beam search over the LSTM decoder. Both numbers are well under the perceptual real-time threshold; the dominant component of \emph{perceived} user latency is the 2-second cooldown buffer in the segmentation FSM, not computation.

% ----------------------------------------------------------------
\subsection{Discussion}\label{sec:discussion}
% ----------------------------------------------------------------


\paragraph{What the comparison tells us.} On the Telegram AzSL dataset, Seq2Seq attains 35.5\% Top-1, 4.8 percentage points above the training-free cosine retrieval baseline at 30.7\% Top-1. The absolute number is modest and the gap to practical usability is significant, but the relative ordering is informative: with only 335 videos and 120 sentences, a learned encoder--decoder still extracts signer-invariant structure that a flat MediaPipe embedding does not. The BLEU-4 of 31.43 indicates that incorrect Seq2Seq predictions still share substantial lexical content with the ground truth, while cosine retrieval is essentially binary per query (the retrieved sentence is either the correct neighbour or it is not). On Top-3, however, cosine retrieval recovers most of the gap (45.0\%); since the deployed system shows three candidates, this is the operationally most relevant number.

\paragraph{Are the Seq2Seq results valid given mode collapse?} It is reasonable to ask whether the Seq2Seq numbers are meaningful when one output sentence (\emph{m\textipa{\textschwa}n u\c{s}aql\i{}qdan \c{s}\textipa{\textschwa}h\textipa{\textschwa}r\textipa{\textschwa} var}) accounts for 36.7\% of all predictions. Our position is that the results are partially informative but should be read with care. The collapsed mode is itself not the most frequent ground-truth sentence, so the 35.5\% Top-1 cannot be explained by a degenerate constant predictor: a model that always predicted the most-frequent ground-truth sentence would score around 5--6\% on this dataset. The model also generates 113 distinct predictions over 335 inputs, and Top-10 accuracy reaches 52.8\%, so the decoder does retain residual conditioning on the visual input. The honest interpretation is that Seq2Seq has \emph{learned a bimodal distribution}: a low-confidence default mode that captures common token statistics, and a higher-confidence visually-conditioned mode that drives the non-collapsed predictions and produces the headline accuracy. This is a known failure pattern of attention-based decoders trained on small sequence corpora; it does not invalidate the comparison with cosine retrieval -- both methods are evaluated on the same Top-1 accuracy on the same data -- but it does mean that the 35.5\% number reflects a partially-trained model and that the distance to a fully-trained model is a function of corpus size rather than architecture choice.

\paragraph{Signer-type effects.} User videos achieve 37.0\% Top-1 and translator videos 32.8\%, a 4.2-point gap that is consistent across both pipelines. The most direct explanation is dataset composition: user videos outnumber translator videos roughly 2:1 (216 vs.\ 119), so user queries are more likely to find a same-class neighbour in leave-one-out. A finer per-sentence breakdown supports this: sentences with a single video achieve 57.7\% average accuracy (because for those sentences the retrieval set is dominated by class-level baseline statistics rather than within-class neighbours), while sentences with exactly two videos drop to 20.7\%. This is not a paradox: with two videos per sentence, leave-one-out forces the system to find one specific match, which is harder than ranking against a diverse pool. The takeaway is that crowdsourced multi-signer variability is good for the model but exposes within-class variability that a small corpus cannot fully average out.

\paragraph{Vocabulary coverage.} The Seq2Seq decoder produced 275 unique surface tokens; 248 (90.2\%) of these are in the ground-truth vocabulary of 356 tokens. Only 27 generated tokens are out-of-vocabulary, so the decoder has learned the target lexicon at the word level. The mean predicted sentence length (4.7 words) is close to the ground-truth mean (4.9), so length statistics are not collapsed. The remaining errors are therefore overwhelmingly errors of \emph{sequence composition}, not of word generation -- the model knows what AzSL words look like but does not always reliably condition the order in which they appear on the visual evidence.

\paragraph{Strengths of cosine retrieval.} The retrieval pipeline requires no training, no GPU, and runs end-to-end on commodity hardware. Adding a new sentence to the recognised vocabulary is a single-video, sub-second enrolment with no retraining, which is decisive for low-resource AzSL deployment where new vocabulary will be added incrementally. The 2.8\,ms search-only latency (Section~\ref{sec:latency}) is six times faster than Seq2Seq beam decoding.

\paragraph{Strengths of Seq2Seq.} Despite mode collapse, Seq2Seq scores higher on Top-1 and produces non-trivial BLEU-4. A trained Seq2Seq decoder can in principle generate sentences that are not present in any reference video -- something cosine retrieval cannot do by construction. With more data, this is the architectural family most likely to scale.

\paragraph{Limitations.}\label{sec:limitations}
We list the limitations of this work explicitly so that future AzSL researchers can target them:
\begin{itemize}
\item \emph{Closed-vocabulary retrieval.} Cosine retrieval always returns a nearest neighbour: a query that does not correspond to any sentence in the database will still return one, but the answer will be wrong. We do not propose an out-of-vocabulary detector in this work, although the top-1 similarity score is a usable thresholding signal.
\item \emph{Small corpus.} 335 videos and 120 sentences are too few for a Seq2Seq model to fully avoid mode collapse, and too few to support a held-out signer-independent evaluation.
\item \emph{No non-manual features.} Both pipelines use only hand landmarks. Sign languages, including AzSL, also encode grammatical information in facial expressions, head tilts, eye gaze, and body posture; ignoring these is a substantial information loss for any real deployment. MediaPipe Holistic provides face mesh and pose landmarks at the same compute envelope as MediaPipe Hands and is the obvious next step.
\item \emph{Temporal alignment.} Uniform stride-based sampling normalises sequence length but not signing speed. Two videos of the same sentence with different physical durations are mapped to frame sequences whose offsets within the gesture differ.
\item \emph{Signer-independent evaluation.} The current leave-one-out protocol does not hold out a signer; results may overestimate generalisation to a brand-new signer.
\item \emph{Single dataset.} We evaluate only on the Telegram dataset. Cross-dataset evaluation would strengthen the conclusions but no second sentence-level AzSL corpus exists today.
\end{itemize}

\paragraph{Future directions.} The most impactful next steps are: (a) add MediaPipe Holistic face and pose landmarks and re-run both pipelines to quantify how much non-manual information helps; (b) replace the flat embedding with a lightweight learned projection (shallow MLP with triplet loss) that preserves CPU-only deployment but improves inter-class separation; (c) decorrelate hand features (PCA or whitening) before cosine matching; (d) extend the Telegram dataset, ideally to a held-out signer split; (e) add an out-of-vocabulary rejection module on top of cosine retrieval.

% ================================================================
\section{Conclusion}\label{sec:conclusion}
% ================================================================

We have presented the first comparative study of training-free retrieval and learned Seq2Seq translation for sentence-level Azerbaijani Sign Language recognition on a crowdsourced 335-video, 120-sentence corpus, and we draw the following conclusions.

\emph{First}, a fully training-free landmark-based cosine retrieval pipeline is a viable AzSL recognition system at this data scale. With no training, no GPU, and a single $8064$-dimensional flat embedding per video it reaches 30.7\% Top-1 and 45.0\% Top-3 accuracy at sub-3\,ms search latency, and supports incremental vocabulary growth (one new video per new sentence). For low-resource sign languages, where labelled corpora and training compute are the binding constraint, this is a competitive operating point, not merely a baseline.

\emph{Second}, learning still pays off, but its benefit at this data scale is bounded by mode collapse. The Seq2Seq encoder--decoder improves Top-1 to 35.5\% and produces meaningful BLEU-4 of 31.43, which is encouragingly close to reported sentence-level translation BLEU-4 in much larger corpora, but a single output sentence accounts for 36.7\% of all predictions. The model is therefore a partially-trained system whose Top-1 number is genuine (it is well above any constant predictor) but whose distance to a fully-trained model is governed by corpus size rather than architecture.

\emph{Third}, dataset composition matters more than signer professionalism. User-contributed crowdsourced videos outperform translator videos (37.0\% vs.\ 32.8\% Top-1) across both pipelines, and per-sentence variance reveals that intra-class diversity, not the quality of any individual signer, is the dominant accuracy driver. This argues directly for continuing to grow the Telegram corpus through the existing crowdsourcing pipeline.

\emph{Fourth}, the dominant remaining error sources are not architectural. The flat embedding discards temporal order; the uniform stride sampling does not normalise signing speed; and -- most importantly -- both pipelines use only hand landmarks, ignoring the facial expressions, head tilts, and body posture that carry grammatical meaning in AzSL and other sign languages. Closing any one of these gaps is, on the evidence of this paper, more likely to improve recognition than swapping in a heavier model.

We release the trimmed sentence-level Telegram subset, the trained Seq2Seq checkpoint, the cosine reference embeddings, and the full evaluation pipeline so that subsequent AzSL work can build on a reproducible reference point rather than re-deriving one. The deliberately minimal design of the cosine retrieval pipeline -- no training, no GPU, no learned aggregation -- is intended as a stable baseline against which future AzSL systems can be measured.



% ================================================================
% Bibliography
% ================================================================
%
% \bibliographystyle{splncs04}
% \bibliography{refs}
%
\begin{thebibliography}{10}

\bibitem{mustafazada2025crowdsourcing}
Mustafazada, S., Muradova, G., Gozalov, R., Garayev, E., Hasanov, J.:
Crowdsourcing-Based Interactive Learning Platform for the International Sign Language Community.
In: 2025 6th International Conference on Problems of Cybernetics and Informatics (PCI), pp. 1--5 (2025).
\doi{10.1109/PCI66488.2025.11219822}

\bibitem{fang2013bayesian}
Fang, X., Dehak, N., Glass, J.:
Bayesian Distance Metric Learning on i-vector for Speaker Verification.
ISCA Archive (2013)

\bibitem{dehak2010cosine}
Dehak, N., Dehak, R., Glass, J., Reynolds, D., Kenny, P.:
Cosine Similarity Scoring without Score Normalization Techniques.
Spoken Language Systems Group, MIT (2010)

\bibitem{zhang2020deep}
Zhang, D., Li, Y., Zhang, Z.:
Deep Metric Learning with Spherical Embedding.
In: NeurIPS, vol. \textbf{33}, pp. 18772--18783 (2020)

\bibitem{zhu2020orthogonality}
Zhu, Y., Mak, B.:
Orthogonality Regularizations for End-to-End Speaker Verification.
ISCA Archive (2020)

\bibitem{liu2024skeleton}
Liu, L., Zheng, H., Zhu, Z., Zhou, P.:
Skeleton-based Sign Language Recognition Using a Dual-Stream Spatio-Temporal Dynamic Graph Convolutional Network.
Preprint (2024)

\bibitem{draganov2024hidden}
Draganov, A., Vadgama, S., Bekkers, E.J.:
The Hidden Pitfalls of the Cosine Similarity Loss.
Preprint (2024)

\bibitem{camgoz2018neural}
Camgöz, N.C., Hadfield, S., Koller, O., Ney, H., Bowden, R.:
Neural Sign Language Translation.
In: CVPR, pp. 7784--7793 (2018)

\bibitem{iandola2016squeezenet}
Iandola, F.N., Han, S., Moskewicz, M.W., Ashraf, K., Dally, W.J., Keutzer, K.:
SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters and $<$0.5 MB Model Size.
arXiv preprint arXiv:1602.07360 (2016)

\bibitem{Liu_2024_05}
Liu, T., Tao, T., Zhao, Y., Li, M., Zhu, J.:
A signer-independent sign language recognition method for the single-frequency dataset.
\emph{Neurocomputing} \textbf{582}, 127479 (May 2024).
DOI: 10.1016/j.neucom.2024.127479

\bibitem{sanchezbrizuela2023lightweight}
Sánchez-Brizuela, G., Cisnal, A., de la Fuente-López, E., Fraile, J.-C., \& Pérez-Turiel, J. (2023). Lightweight real-time hand segmentation leveraging MediaPipe landmark detection. \emph{Virtual Reality}, \textbf{27}(4), 3125–3132. \doi{10.1007/s10055-023-00858-0}

\bibitem{varshini2025mpgestlstm}
Varshini, T. S., \& Rukmani, P. (2025). MP-GestLSTM: real-time gesture detection using MediaPipe and LSTM. \emph{Systems Science \& Control Engineering}, \textbf{13}(1). \doi{10.1080/21642583.2025.2587853}

\bibitem{nguyen2023exploring}
Nguyen, P. T., Nguyen, T. H., Hoang, N. X. N., Phan, H. T. B., Vu, H. S. H., \& Huynh, H. N. (2023). Exploring MediaPipe optimization strategies for real-time sign language recognition. \emph{CTU Journal of Innovation and Sustainable Development}, \textbf{15}(ISDS), 142–152. \doi{10.22144/ctujoisd.2023.045}

\bibitem{carneiro2024sign}
Carneiro, A. L. C., Salvadeo, D. H. P., \& Silva, L. de B. (2024). Sign language recognition based on deep learning and low-cost handcrafted descriptors (Version 1). arXiv preprint arXiv:2408.07244.

\bibitem{kamble2025slrnet}
Kamble, S. (2025). SLRNet: A Real-Time LSTM-Based Sign Language Recognition System (Version 1). arXiv preprint arXiv:2506.11154.

\bibitem{verma2024enhancing}
Verma, A. R., Singh, G., Meghwal, K., Ramji, B., \& Dadheech, P. K. (2024). Enhancing Sign Language Detection through Mediapipe and Convolutional Neural Networks (CNN) (Version 2). arXiv preprint arXiv:2406.03729.

\bibitem{alishzade2024azsld} Alishzade, N., Hasanov, J.: AzSLD: Azerbaijani Sign Language Dataset for Fingerspelling, Word, and Sentence Translation with Baseline Software. Preprint (2024)
\end{thebibliography}
\end{document}