//
//  YoutuLLM.swift
//  mlx-swift-lm
//
//  Created by Molly Sophia on 2026/4/23.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/youtu_llm.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct YoutuLLMConfiguration: Codable, Sendable {
    var modelType: String = "youtu_llm"
    var vocabSize: Int = 128256
    var hiddenSize: Int = 2048
    var intermediateSize: Int = 6144
    var numHiddenLayers: Int = 32
    var numAttentionHeads: Int = 16
    var numKeyValueHeads: Int = 16
    var kvLoraRank: Int = 512
    var qLoraRank: Int? = 1536
    var qkRopeHeadDim: Int = 64
    var vHeadDim: Int = 128
    var qkNopeHeadDim: Int = 128
    var maxPositionEmbeddings: Int = 131072
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 1_600_000
    var ropeTraditional: Bool = true
    var ropeInterleave: Bool? = nil
    var ropeScaling: [String: StringOrNumber]? = nil
    var ropeParameters: [String: StringOrNumber]? = nil
    var attentionBias: Bool = false
    var mlpBias: Bool = false
    var tieWordEmbeddings: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeInterleave = "rope_interleave"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "youtu_llm"
        self.vocabSize =
            try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 128256
        self.hiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        self.numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 32
        self.numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 16
        self.kvLoraRank =
            try container.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
        self.qLoraRank =
            try container.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 1536
        self.qkRopeHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        self.vHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
        self.qkNopeHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 128
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.rmsNormEps =
            try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeParameters =
            try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)
        // prefer top-level rope_theta, else from rope_parameters
        if let rt = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            self.ropeTheta = rt
        } else if let rt = self.ropeParameters?["rope_theta"]?.asFloat() {
            self.ropeTheta = rt
        } else {
            self.ropeTheta = 1_600_000
        }
        // prefer rope_traditional; else fall back to rope_interleave
        if let rt = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) {
            self.ropeTraditional = rt
        } else if let ri = try container.decodeIfPresent(Bool.self, forKey: .ropeInterleave) {
            self.ropeInterleave = ri
            self.ropeTraditional = ri
        } else {
            self.ropeTraditional = true
        }
        self.ropeScaling =
            try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.mlpBias =
            try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }
}

private class YoutuLLMAttention: Module {
    var config: YoutuLLMConfiguration
    var hiddenSize: Int
    var numHeads: Int
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var qLoraRank: Int?
    var qkRopeHeadDim: Int
    var kvLoraRank: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var qHeadDim: Int
    var scale: Float

    let rope: RoPELayer
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: Module  // Can be MultiLinear or QuantizedMultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOut: Module  // Can be MultiLinear or QuantizedMultiLinear

    init(config: YoutuLLMConfiguration) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.ropeTheta = config.ropeTheta
        self.qLoraRank = config.qLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim

        self.scale = pow(Float(qHeadDim), -0.5)

        if let qLoraRank = qLoraRank {
            self._qAProj.wrappedValue = Linear(
                hiddenSize, qLoraRank, bias: config.attentionBias
            )
            self._qALayerNorm.wrappedValue = RMSNorm(
                dimensions: qLoraRank, eps: config.rmsNormEps)
            self._qBProj.wrappedValue = Linear(
                qLoraRank, numHeads * qHeadDim, bias: false
            )
        } else {
            self._qProj.wrappedValue = Linear(hiddenSize, numHeads * qHeadDim, bias: false)
        }

        self._kvAProjWithMqa.wrappedValue = Linear(
            hiddenSize,
            kvLoraRank + qkRopeHeadDim,
            bias: config.attentionBias
        )
        self._kvALayerNorm.wrappedValue = RMSNorm(
            dimensions: kvLoraRank, eps: config.rmsNormEps)

        self._embedQ.wrappedValue = MultiLinear(
            inputDims: qkNopeHeadDim,
            outputDims: kvLoraRank,
            numHeads: numHeads
        )
        self._unembedOut.wrappedValue = MultiLinear(
            inputDims: kvLoraRank,
            outputDims: vHeadDim,
            numHeads: numHeads
        )

        self._oProj.wrappedValue = Linear(
            numHeads * vHeadDim, hiddenSize, bias: config.attentionBias)

        self.rope = initializeRope(
            dims: qkRopeHeadDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: config.ropeScaling ?? config.ropeParameters,
            maxPositionEmbeddings: maxPositionEmbeddings)
    }

    private func callMultiLinear(_ module: Module, _ x: MLXArray) -> MLXArray {
        if let ml = module as? MultiLinear {
            return ml(x)
        } else if let q = module as? QuantizedMultiLinear {
            return q(x)
        } else {
            fatalError("embed_q/unembed_out must be MultiLinear or QuantizedMultiLinear")
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q: MLXArray
        if qLoraRank == nil {
            q = self.qProj!(x)
        } else {
            q = self.qBProj!(self.qALayerNorm!(self.qAProj!(x)))
        }
        q = q.reshaped(B, L, self.numHeads, self.qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = splitQ[0]
        var qPe = splitQ[1]

        var compressedKv = self.kvAProjWithMqa(x)
        let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitCompressedKv[0]
        var kPe = splitCompressedKv[1]
        kPe = kPe.reshaped(B, L, 1, self.qkRopeHeadDim).transposed(0, 2, 1, 3)
        var kvLatent = kvALayerNorm(compressedKv)

        qPe = applyRotaryPosition(rope, to: qPe, cache: cache)
        kPe = applyRotaryPosition(rope, to: kPe, cache: cache)

        qNope = callMultiLinear(embedQ, qNope)
        kvLatent = expandedDimensions(kvLatent, axis: 1)

        // Cache kv_latent and k_pe separately (DeepSeek-V3 absorbed MLA style).
        var kvLatentCached = kvLatent
        var kPeCached = kPe
        if let cache {
            (kvLatentCached, kPeCached) = cache.update(keys: kvLatent, values: kPe)
        }

        // Fold the RoPE (pe) contribution of Q·K into an additive mask, so SDPA
        // only runs on the kv_lora_rank-wide latent space.
        var peScores = matmul(qPe * scale, kPeCached.swappedAxes(-1, -2))
        if let mask {
            let negInf = MLXArray(Float(-Float.infinity)).asType(peScores.dtype)
            peScores = which(mask, peScores, negInf)
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: qNope,
            keys: kvLatentCached,
            values: kvLatentCached,
            scale: scale,
            mask: .array(peScores)
        )

        output = callMultiLinear(unembedOut, output)
        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return self.oProj(output)
    }
}

private class YoutuLLMMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: YoutuLLMConfiguration) {
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self._upProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self._downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        self.downProj(silu(self.gateProj(x)) * self.upProj(x))
    }
}

private class YoutuLLMDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: YoutuLLMAttention
    @ModuleInfo var mlp: YoutuLLMMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: YoutuLLMConfiguration) {
        self._selfAttn.wrappedValue = YoutuLLMAttention(config: config)
        self.mlp = YoutuLLMMLP(config: config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

public class YoutuLLMModelInner: Module {
    var config: YoutuLLMConfiguration
    var vocabSize: Int
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate var layers: [YoutuLLMDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: YoutuLLMConfiguration) {
        self.config = config
        self.vocabSize = config.vocabSize
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map { _ in
            YoutuLLMDecoderLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(x)

        // Force the array form so the MLA attention can fold it into peScores.
        let maskMode = createAttentionMask(
            h: h, cache: cache?.first, returnArray: true)
        let causalMask: MLXArray?
        switch maskMode {
        case .array(let m): causalMask = m
        case .none, .causal, .arrays: causalMask = nil
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: causalMask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class YoutuLLMModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int]

    var config: YoutuLLMConfiguration
    public var model: YoutuLLMModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: YoutuLLMConfiguration) {
        self.config = config
        // Absorbed-MLA KV cache is MQA-shared across heads (head dim = 1).
        self.kvHeads = Array(repeating: 1, count: config.numHiddenLayers)
        self.model = YoutuLLMModelInner(config: config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights
        if config.tieWordEmbeddings {
            newWeights["lm_head.weight"] = nil
        }

        // Convert kv_b_proj to embed_q and unembed_out
        let numHeads = config.numAttentionHeads
        let headDim = config.qkNopeHeadDim + config.vHeadDim
        for l in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(l).self_attn"
            let wKey = "\(prefix).kv_b_proj.weight"
            guard var v = newWeights[wKey] else { continue }
            newWeights.removeValue(forKey: wKey)

            let isQuantized = newWeights["\(prefix).kv_b_proj.scales"] != nil
            var inferredBits = 0
            var inferredGroupSize = 0

            if isQuantized {
                let scales = newWeights.removeValue(forKey: "\(prefix).kv_b_proj.scales")!
                let biases = newWeights.removeValue(forKey: "\(prefix).kv_b_proj.biases")!
                // Infer bits and group size
                inferredBits = (v.dim(-1) * 32) / config.kvLoraRank
                inferredGroupSize = config.kvLoraRank / scales.dim(-1)
                v = dequantized(
                    v, scales: scales, biases: biases,
                    groupSize: inferredGroupSize, bits: inferredBits
                )
            }

            v = v.reshaped(numHeads, headDim, -1)
            var wk = v[0..., ..<config.qkNopeHeadDim, 0...].swappedAxes(-1, -2)
            var wv = v[0..., config.qkNopeHeadDim..., 0...]

            // Make contiguous
            wk = contiguous(wk)
            wv = contiguous(wv)

            if isQuantized {
                let (qWk, qWkScales, qWkBiases) = MLX.quantized(
                    wk, groupSize: inferredGroupSize, bits: inferredBits)
                let (qWv, qWvScales, qWvBiases) = MLX.quantized(
                    wv, groupSize: inferredGroupSize, bits: inferredBits)
                newWeights["\(prefix).embed_q.scales"] = qWkScales
                newWeights["\(prefix).embed_q.biases"] = qWkBiases
                newWeights["\(prefix).unembed_out.scales"] = qWvScales
                newWeights["\(prefix).unembed_out.biases"] = qWvBiases
                wk = qWk
                wv = qWv
            }
            newWeights["\(prefix).embed_q.weight"] = wk
            newWeights["\(prefix).unembed_out.weight"] = wv
        }

        return newWeights.filter { key, _ in
            !key.contains("rotary_emb.inv_freq")
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
