# Aula 3: Controle do Modelo via Aplicação (Geração e Sampling)

Nas aulas anteriores, aprendemos que os modelos são APIs *stateless* e que utilizamos a Engenharia de Prompts para ensinar comportamentos em tempo de execução. Mas como exatamente o modelo escolhe as palavras que formam a resposta? E como podemos controlar a "criatividade" e o tempo de resposta dessa geração através da nossa aplicação Java?

Nesta aula, deixamos a construção de prompts de lado para focar na matemática da geração de tokens, explorando os parâmetros de amostragem (*sampling*), o impacto no tempo de resposta e como a natureza do modelo afeta a confiabilidade do nosso software.

---

## 1. A Natureza Probabilística e a Origem das Alucinações

Para controlar um Large Language Model (LLM), o primeiro passo é entender que a sua geração de texto é fundamentalmente probabilística. 

Dado um texto de entrada, a rede neural do modelo produz um vetor chamado de *logits*, onde cada *logit* corresponde a um token possível do vocabulário do modelo. Esses *logits* puros não são probabilidades reais (podem ser números negativos e não somam 1), então eles passam por uma camada matemática chamada *softmax*, que os converte em uma distribuição de probabilidade.

Essa natureza probabilística significa que, em vez de respostas determinísticas, o modelo "sorteia" o próximo token com base nessas probabilidades. É isso que causa dois dos maiores desafios na construção de chatbots corporativos:

* **Inconsistência:** Se você fizer a mesma pergunta duas vezes, o modelo pode "sortear" caminhos probabilísticos diferentes e gerar duas respostas completamente distintas.
* **Alucinações:** Modelos não têm um banco de dados de fatos; eles são agregações estatísticas. Qualquer combinação de palavras com probabilidade maior que zero pode ser gerada. Se o modelo calcular que uma afirmação falsa é estatisticamente provável no contexto fornecido, ele a gerará com total confiança, resultando em uma alucinação.

---

## 2. Manipulação de Parâmetros de *Sampling* (Amostragem)

Se o modelo apenas escolhesse sempre o token com a maior probabilidade (uma técnica chamada *greedy sampling*), suas respostas seriam consistentes, porém extremamente chatas, repetitivas e previsíveis. Para criar conversas naturais, nós permitimos que o modelo faça um *sampling* (amostragem) entre as opções prováveis.

Nós controlamos as rédeas dessa amostragem manipulando parâmetros na chamada da API. No ecossistema Quarkus LangChain4j, isso é feito de forma totalmente declarativa via `application.properties`, mantendo o código Java limpo.

### Temperature (Temperatura)
A temperatura ajusta os *logits* antes da conversão para probabilidade. 
* **Temperatura Alta (ex: 0.7 a 1.0):** Achata a curva de probabilidade, reduzindo a chance dos tokens mais óbvios e aumentando a chance de tokens raros. O chatbot fica mais criativo e diverso, ideal para redação e *brainstorming*.
* **Temperatura Baixa (ex: 0.0 a 0.2):** Acentua a curva. O modelo escolhe quase sempre a resposta mais óbvia. É a configuração recomendada para tarefas de classificação, extração de dados e RAG, onde a precisão e o determinismo importam mais que a criatividade.

### Top-k e Top-p (Nucleus Sampling)
Para evitar que o modelo escolha tokens absurdos durante a amostragem, aplicamos filtros matemáticos:
* **Top-k:** Filtra a lista para considerar apenas os *K* tokens mais prováveis antes de fazer a amostragem (ex: k=40). Isso reduz o custo computacional sem sacrificar muita diversidade.
* **Top-p:** Filtra dinamicamente os tokens cuja probabilidade acumulada atinja um valor *P* (ex: 0.9 ou 90%). Isso permite que o modelo considere mais palavras quando está incerto, e menos palavras quando a próxima palavra é muito óbvia.

**Configuração Declarativa no Quarkus:**

```properties
# ==========================================
# Perfil 1: Criativo (Ideal para Brainstorming e Redação)
# Funil Largo: Permite alta diversidade de palavras
# ==========================================
quarkus.langchain4j.ollama.criativo.chat-model.model-id=gpt-oss:20b-cloud
quarkus.langchain4j.ollama.criativo.chat-model.temperature=0.9
# Olha para um conjunto grande de possibilidades (valores comuns variam de 50 a 500)
quarkus.langchain4j.ollama.criativo.chat-model.top-k=50
# Considera os tokens cuja probabilidade acumulada chega a 95%
quarkus.langchain4j.ollama.criativo.chat-model.top-p=0.95

# ==========================================
# Perfil 2: Determinístico (Ideal para RAG e Extração de Dados)
# Funil Estrangulado: Corta qualquer aleatoriedade
# ==========================================
quarkus.langchain4j.ollama.deterministico.chat-model.model-id=gpt-oss:20b-cloud
# Temperatura 0.0 força o modelo a agir de forma gananciosa (Greedy Sampling)
quarkus.langchain4j.ollama.deterministico.chat-model.temperature=0.0
# Top-K 1 torna o texto extremamente previsível, olhando apenas para o token mais provável
quarkus.langchain4j.ollama.deterministico.chat-model.top-k=1
# Com o funil já restrito pelo Top-K, um Top-P baixo (ex: 10%) cimenta a restrição
quarkus.langchain4j.ollama.deterministico.chat-model.top-p=0.10
```

```java
package org.acme.ai;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import io.quarkiverse.langchain4j.RegisterAiService;

// Serviço apontando para as configurações "criativo"
@RegisterAiService(modelName = "criativo")
public interface GeradorCriativoService {
    
    @SystemMessage("Você é um publicitário genial e fora da caixa.")
    @UserMessage("Escreva um slogan de apenas uma frase para uma loja de roupas para gatos chamada 'Kitty Vogue': {contexto}")
    String gerarSlogan(String contexto);
}
```

```java
package org.acme.ai;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import io.quarkiverse.langchain4j.RegisterAiService;

// Serviço apontando para as configurações "deterministico"
@RegisterAiService(modelName = "deterministico")
public interface AnalisadorDeterministicoService {
    
    @SystemMessage("Você é um publicitário genial e fora da caixa.")
    @UserMessage("Escreva um slogan de apenas uma frase para uma loja de roupas para gatos chamada 'Kitty Vogue': {contexto}")
    String gerarSlogan(String contexto);
}
```

```java
package org.acme.api;

import org.acme.ai.GeradorCriativoService;
import org.acme.ai.AnalisadorDeterministicoService;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;

@Path("/aula3")
@Produces(MediaType.TEXT_PLAIN)
public class SamplingDemoResource {

    @Inject
    GeradorCriativoService criativoService;

    @Inject
    AnalisadorDeterministicoService deterministicoService;

    @GET
    @Path("/comparar-sampling")
    public String compararSampling() {
        
        String contexto = "Roupas elegantes e confortáveis.";

        // Invocamos ambos os modelos com o EXATO mesmo prompt
        String respostaCriativa = criativoService.gerarSlogan(contexto);
        String respostaDeterministica = deterministicoService.gerarSlogan(contexto);

        return """
               === DEMONSTRAÇÃO DE SAMPLING ===
               
               [Modelo Criativo (Temp: 0.9, Top-P: 0.9)]
               %s
               
               [Modelo Determinístico (Temp: 0.0, Top-K: 1)]
               %s
               
               ================================
               Dica para os alunos: Chamem este endpoint 3 vezes seguidas. 
               A resposta do modelo criativo vai mudar (Inconsistência controlada). 
               A resposta do modelo determinístico será exatamente a mesma todas as vezes (Greedy Sampling).
               """.formatted(respostaCriativa, respostaDeterministica);
    }
}
```

---

## 3. Streaming de Respostas e a Métrica TTFT

Como vimos, a geração de texto em LLMs é *autorregressiva*, ou seja, o modelo gera uma palavra de cada vez, dependendo da palavra anterior. O processamento disso é severamente limitado pela largura de banda da memória da GPU. 

Se um usuário pedir um resumo de 1.000 tokens e o modelo levar 100 milissegundos para gerar cada token, o usuário ficará olhando para uma tela em branco por mais de 1 minuto antes de ver a resposta. Essa latência total destrói a experiência do usuário.

A solução arquitetural é focar no **TTFT (Time to First Token)**. Em vez de esperar a geração completa da resposta (o que chamamos de comportamento blocante), utilizamos o recurso de **Streaming de Respostas**, onde a API transmite cada token pela rede assim que ele é gerado.

No Java moderno com Quarkus LangChain4j, lidamos com isso usando o paradigma de programação reativa, retornando um `Multi<String>` (do framework Mutiny) no nosso AiService.

**Serviço de IA Reativo:**

```java
package org.acme.ai;

import dev.langchain4j.service.UserMessage;
import io.quarkiverse.langchain4j.RegisterAiService;
import io.smallrye.mutiny.Multi;

@RegisterAiService
public interface StreamingChatService {

    @UserMessage("Explique detalhadamente a arquitetura de microserviços: {pergunta}")
    // O retorno em Multi<String> diz ao framework para abrir um stream reativo
    Multi<String> chatReativo(String pergunta);
}
```

**Resource JAX-RS Expondo Server-Sent Events (SSE):**

```java
package org.acme.api;

import org.acme.ai.StreamingChatService;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import org.jboss.resteasy.reactive.RestStreamElementType;
import io.smallrye.mutiny.Multi;

@Path("/chatbot")
public class StreamingResource {

    @Inject
    StreamingChatService chatService;

    @GET
    @Path("/stream")
    @Produces(MediaType.SERVER_SENT_EVENTS) // Mantém a conexão HTTP aberta
    @RestStreamElementType(MediaType.TEXT_PLAIN)
    public Multi<String> streamResposta(String pergunta) {
        // Reduz o TTFT para milissegundos, enviando pedaços do texto ao frontend em tempo real
        return chatService.chatReativo(pergunta);
    }
}
```

---

## 4. Condições de Parada (Stopping Conditions)

Um modelo autorregressivo pode, teoricamente, continuar gerando tokens infinitamente. Respostas excessivamente longas gastam os créditos da sua API (pois você paga ou consome processamento por token gerado) e também testam a paciência do usuário.

Para aplicar engenharia defensiva na camada da aplicação, nós impomos **Condições de Parada (Stopping Conditions)** na chamada da API. 

A abordagem mais simples e mais utilizada em sistemas de produção é definir um limite máximo absoluto de tokens de saída (*Max Tokens*). Ao definir esse parâmetro, garantimos previsibilidade no custo computacional. No Quarkus, adicionamos apenas uma propriedade no arquivo `application.properties`:

```properties
# O modelo interromperá a geração imediatamente ao atingir 500 tokens
quarkus.langchain4j.ollama.chat-model.max-tokens=500
```

**O Trade-off de Sistemas Corporativos:**
A principal desvantagem da condição de parada abrupta é que a saída será cortada no meio de uma frase. Se o seu chatbot estiver programado para retornar uma estrutura de dados rigorosa, como um objeto JSON, e a geração parar precocemente, você receberá um JSON malformado (sem as chaves de fechamento `}`), o que causará o travamento do *Parser* no backend Java. Portanto, é crucial estimar adequadamente o limite de tokens para as tarefas da sua aplicação corporativa!
