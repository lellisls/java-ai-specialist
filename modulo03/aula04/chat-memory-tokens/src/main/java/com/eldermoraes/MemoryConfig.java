package com.eldermoraes;

import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.memory.chat.TokenWindowChatMemory;
import dev.langchain4j.model.TokenCountEstimator;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.Produces;

@ApplicationScoped
public class MemoryConfig {

    @Inject
    TokenCountEstimator  tokenCountEstimator;

    @Produces
    public ChatMemoryProvider chatMemoryProvider() {
        return memoryId -> TokenWindowChatMemory.builder()
                .id(memoryId)
                .maxTokens(1000, tokenCountEstimator)
                .build();
    }
}
