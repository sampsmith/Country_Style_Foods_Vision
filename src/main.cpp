#include "main_application.h"
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Country Style Dough Inspector - High Performance Edition" << std::endl;
    std::cout << "Target: <10ms per frame processing" << std::endl;
    std::cout << "===================================================" << std::endl;
    
    country_style::MainApplication app;
    
    if (!app.initialize()) {
        std::cerr << "Failed to initialize application" << std::endl;
        return -1;
    }
    
    app.run();
    app.shutdown();
    
    return 0;
}
