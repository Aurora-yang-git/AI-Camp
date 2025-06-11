import {
  LanguageModelV1,
  extractReasoningMiddleware,
  generateText,
  wrapLanguageModel,
} from "ai";
import { respData, respErr } from "@/lib/resp";

import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { deepseek } from "@ai-sdk/deepseek";
import { openai } from "@ai-sdk/openai";
import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const { prompt, provider, model } = await req.json();
    if (!prompt || !provider) {
      return respErr("invalid params");
    }
    
    // Hugging Face 处理逻辑
    if (provider === 'huggingface') {
      const huggingfaceModel = model || 'google/gemma-7b'; // 默认使用正确的Google Gemma 7B模型
      
      // 定义在try块外部，以便在catch中也能访问
      const controller = new AbortController();
      let timeoutId: NodeJS.Timeout | null = setTimeout(() => controller.abort(), 60000); // 增加超时到60秒
      
      try {
        console.log(`尝试连接Hugging Face API，模型：${huggingfaceModel}...`);
        
        const response = await fetch(`https://api-inference.huggingface.co/models/${huggingfaceModel}`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
            'Content-Type': 'application/json'
          },
          signal: controller.signal, // 添加AbortController信号
          body: JSON.stringify({
            inputs: prompt,
            parameters: {
              max_new_tokens: 1024,
              temperature: 0.7,
              top_p: 0.95,
              do_sample: true
            }
          })
        });
        clearTimeout(timeoutId); // 清理超时定时器
        
        if (!response.ok) {
          console.error(`Hugging Face API error: ${response.statusText}`);
          return respErr(`Hugging Face API error: ${response.statusText}`);
        }
        
        const result = await response.json();
        const responseText = result[0].generated_text.replace(prompt, '').trim();
        
        return respData({
          text: responseText,
          reasoning: null
        });
      } catch (error) {
        clearTimeout(timeoutId); // 确保在错误情况下也清理超时定时器
        console.error('Hugging Face API error:', error);
        return respErr(`Hugging Face API error: ${error instanceof Error ? error.message : '未知错误'}`);
      }
    }
    
    // 原有的处理逻辑（非 Hugging Face 提供商）
    if (!model) {
      return respErr("invalid params: model is required");
    }
    
    let textModel: LanguageModelV1;

    // 处理其他提供商
    switch (provider) {
      case "openai":
        textModel = openai(model);
        break;
      case "deepseek":
        // 使用 createOpenAICompatible 创建 DeepSeek 客户端
        const deepseekClient = createOpenAICompatible({
          name: "deepseek",
          apiKey: process.env.DEEPSEEK_API_KEY || "",
          baseURL: process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com"
        });
        textModel = deepseekClient(model);
        break;
      case "openrouter":
        const openrouter = createOpenRouter({
          apiKey: process.env.OPENROUTER_API_KEY,
        });
        textModel = openrouter(model);

        if (model === "deepseek/deepseek-r1") {
          const enhancedModel = wrapLanguageModel({
            model: textModel,
            middleware: extractReasoningMiddleware({
              tagName: "think",
            }),
          });
          textModel = enhancedModel;
        }
        break;
      case "siliconflow":
        const siliconflow = createOpenAICompatible({
          name: "siliconflow",
          apiKey: process.env.SILICONFLOW_API_KEY,
          baseURL: process.env.SILICONFLOW_BASE_URL,
        });
        textModel = siliconflow(model);

        if (model === "deepseek-ai/DeepSeek-R1") {
          const enhancedModel = wrapLanguageModel({
            model: textModel,
            middleware: extractReasoningMiddleware({
              tagName: "reasoning_content",
            }),
          });
          textModel = enhancedModel;
        }
        break;
      default:
        return respErr("invalid provider");
    }

    // 使用 AI SDK 生成文本
    const { reasoning, text, warnings } = await generateText({
      model: textModel,
      prompt: prompt,
    });

    if (warnings && warnings.length > 0) {
      console.log("gen text warnings:", provider, warnings);
      return respErr("gen text failed");
    }

    return respData({
      text: text,
      reasoning: reasoning,
    });
  } catch (err) {
    console.log("gen text failed:", err);
    return respErr("gen text failed");
  }
}
