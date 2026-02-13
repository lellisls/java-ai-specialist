package com.eldermoraes;

import jakarta.annotation.PostConstruct;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;

@Path("star-wars")
@Consumes(MediaType.TEXT_PLAIN)
@Produces(MediaType.TEXT_PLAIN)
public class StarWarsResource {

    @Inject
    StarWarsAssistant assistant;

    @POST
    public String chat(String userMessage) {
        return assistant.chat(userMessage);
    }
}
