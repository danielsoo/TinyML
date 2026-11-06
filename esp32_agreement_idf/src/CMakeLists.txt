idf_component_register(SRCS "main.cc" "model_data.cc" INCLUDE_DIRS "."
                       REQUIRES driver esp_timer spi_flash)
