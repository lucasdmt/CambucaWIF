#include <cstring>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cassert>
#include <pthread.h>
#include <fstream>
#include "GPU/GPUSecp.h"
#include "CPU/SECP256k1.h"
#include "CPU/HashMerge.cpp"
#include <sys/resource.h>
#include <chrono>

#include <cmath> //pow
#include <getopt.h>

#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_RESET   "\033[0m"


bool lerChaveDeArquivo(const char* nomeArquivo, char* strCPU){
   std::ifstream inputFile(nomeArquivo);
   if (!inputFile){
      std::cerr << "Error opening file " << nomeArquivo << std::endl;
      return false;
   }

   std::string line;
   if (!std::getline(inputFile, line)){
      std::cerr << "Error reading line from file " << nomeArquivo << std::endl;
      return false;
   }

   if (line.length() != 64){
      std::cerr << "Error: the file line must contain exactly 64 characters." << std::endl;
      return false;
   }

   strncpy(strCPU, line.c_str(), 64);
   strCPU[64] = '\0';
   std::cout << "Key loaded from file" << std::endl;
   return true;
}


bool validarChaveWIF(const char* strCPU, bool permitir0)
{
    size_t len = strlen(strCPU);

    if (len != 51 && len != 52) {
        std::cerr << "Erro: WIF deve ter 51 ou 52 caracteres" << std::endl;
        return false;
    }

    const char* base58 =
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    for (size_t i = 0; i < len; i++) {
        char c = strCPU[i];

        if (permitir0 && c == '0')
            continue;

        if (strchr(base58, c) == nullptr) {
            std::cerr << "Erro: caractere inválido na posição "
                      << i << ": '" << c << "'" << std::endl;
            return false;
        }
    }

    return true;
}

/*
bool hexStringToLittleEndian(const std::string& strCPU, uint8_t* privKeyCPU){
    // converte a string hex para uint8_t[32] em little-endian
    for (int i = 0; i < 32; i++) {
        // Lê os caracteres hex *do fim para o início* (inverte a ordem dos bytes)
        int posChar = (31 - i) * 2;  // Começa do último par de caracteres
        char highChar = strCPU[posChar];
        char lowChar  = strCPU[posChar + 1];

        // converte para nibbles e combina em um byte
        uint8_t highNibble =
            (highChar >= '0' && highChar <= '9') ? highChar - '0' :
            (highChar >= 'a' && highChar <= 'f') ? highChar - 'a' + 10 :
            (highChar >= 'A' && highChar <= 'F') ? highChar - 'A' + 10 :
            0; // qualquer coisa fora 0-9, a-f, A-F (ex: 'x') vira 0

        uint8_t lowNibble =
            (lowChar >= '0' && lowChar <= '9') ? lowChar - '0' :
            (lowChar >= 'a' && lowChar <= 'f') ? lowChar - 'a' + 10 :
            (lowChar >= 'A' && lowChar <= 'F') ? lowChar - 'A' + 10 :
            0;

        privKeyCPU[i] = (highNibble << 4) | lowNibble;
    }

    return true;
}
*/

void loadInputHash(uint64_t *inputHashBufferCPU) {
   std::cout << "Loading hash buffer from file: " << NAME_HASH_BUFFER << std::endl;

   FILE *fileSortedHash = fopen(NAME_HASH_BUFFER, "rb");
   if (fileSortedHash == NULL)
   {
      printf("Error: not able to open input file: %s\n", NAME_HASH_BUFFER);
      exit(1);
   }

   fseek(fileSortedHash, 0, SEEK_END);
   long hashBufferSizeBytes = ftell(fileSortedHash);
   long hashCount = hashBufferSizeBytes / SIZE_LONG;
   rewind(fileSortedHash);

   if (hashCount != COUNT_INPUT_HASH) {
      printf("ERROR - Constant COUNT_INPUT_HASH is %d, but the actual hashCount is %lu \n", COUNT_INPUT_HASH, hashCount);
      exit(-1);
   }

   size_t size = fread(inputHashBufferCPU, 1, hashBufferSizeBytes, fileSortedHash);
   fclose(fileSortedHash);

   std::cout << "loadInputHash " << NAME_HASH_BUFFER << " finished!" << std::endl;
   std::cout << "hashCount: " << hashCount << ", hashBufferSizeBytes: " << hashBufferSizeBytes << std::endl;
}

void loadGTable(uint8_t *gTableX, uint8_t *gTableY) {
   std::cout << "loadGTable started" << std::endl;

   Secp256K1 *secp = new Secp256K1();
   secp->Init();

   for (int i = 0; i < NUM_GTABLE_CHUNK; i++)
   {
      for (int j = 0; j < NUM_GTABLE_VALUE - 1; j++)
      {
         int element = (i * NUM_GTABLE_VALUE) + j;
         Point p = secp->GTable[element];
         for (int b = 0; b < 32; b++) {
            gTableX[(element * SIZE_GTABLE_POINT) + b] = p.x.GetByte64(b);
            gTableY[(element * SIZE_GTABLE_POINT) + b] = p.y.GetByte64(b);
         }
      }
   }

   std::cout << "loadGTable finished!" << std::endl;
}


void salvarPosicoes(const char *strCPU, int *posicoesCPU, int *totalPosicoesCPU) {
    *totalPosicoesCPU = 0;
    for (int i = 0; strCPU[i] != '\0'; i++) {
        if (strCPU[i] == '0') {
            posicoesCPU[*totalPosicoesCPU] = i;
            (*totalPosicoesCPU)++;
        }
    }
    printf("Total de posições '0' encontradas:%d\n", *totalPosicoesCPU);
    printf("0 nas posições no array:");
    for (int i = 0; i < *totalPosicoesCPU; i++) {
        printf("%d ", posicoesCPU[i]); 
    }
    printf("      ");
    printf("0 nas posições na chave:");
    for (int i = 0; i < *totalPosicoesCPU; i++) {
        printf("%d ", posicoesCPU[i]+1); 
    }
    printf("\n");
}

void printProgressBar(double progress)
{
    const int barWidth = 30;

    printf("[");
    int pos = (int)(barWidth * progress);

    for (int i = 0; i < barWidth; i++)
    {
        if (i < pos) printf("█");
        else printf("░");
    }
    printf("]");
}
void printStrCPU(const char *strCPU, const int *posicoesCPU, int totalPosicoesCPUtemp){
   int mapa[52] = {0};

   // marca as posições que devem ficar vermelhas
   for (int j = 0; j < totalPosicoesCPUtemp; j++)
   {
      if (posicoesCPU[j] >= 0 && posicoesCPU[j] < 64)
         mapa[posicoesCPU[j]] = 1;
   }

   // imprime a chave
   printf(" ");
   for (int i = 0; i < 52; i++)
   {
      if (mapa[i])
         printf(COLOR_RED "%c" COLOR_RESET, strCPU[i]);
      else
         printf("%c", strCPU[i]);
   }
}

void startSecp256k1ModeWIFX(uint8_t * gTableXCPU, uint8_t * gTableYCPU, uint64_t * inputHashBufferCPU, const char* WIFprivKeyCPU) {

   printf("Modo WIF X\n");

   /*printf("privkey cpu hex: ");
   for (int i = 0; i < 32; ++i) {
      printf("%02X", (unsigned)privKeyCPU[i]);
   }printf("\n");*/

   int posicoesCPU[58];
   int totalPosicoesCPUtemp;

   salvarPosicoes(WIFprivKeyCPU, posicoesCPU, &totalPosicoesCPUtemp);

   GPUSecp *gpuSecp = new GPUSecp(
      gTableXCPU,
      gTableYCPU,
      inputHashBufferCPU,
      WIFprivKeyCPU,              
      posicoesCPU,         
      totalPosicoesCPUtemp
   );

   long timeTotal = 0;
   long totalCount = (COUNT_CUDA_THREADS);

    long long possibilidades = 1;
    for(int i = 0; i < totalPosicoesCPUtemp; i++)possibilidades *= 58;

   printf("Chave Parcial char: ");
   printStrCPU(WIFprivKeyCPU, posicoesCPU, totalPosicoesCPUtemp);
   printf("\nPossibilidades por combinação (58^%d): %llu\n", totalPosicoesCPUtemp, possibilidades);


   int itercount = COUNT_CUDA_THREADS * THREAD_MULT;
   long long maxIteration = (possibilidades + itercount - 1) / itercount;
   printf("cada iteração resulta em %d tentativas, resultando em no maximo de %lld iterações \n", itercount, maxIteration);

   auto clockStart = std::chrono::high_resolution_clock::now();
   for (int iter = 0; iter < maxIteration + 1; iter++){
   //for (int iter = 0; iter < 1000000; iter++){

      const auto clockIter1 = std::chrono::high_resolution_clock::now();

      gpuSecp->doIterationSecp256k1Books(iter);
      gpuSecp->doPrintOutput();

      const auto clockIter2 = std::chrono::high_resolution_clock::now();

      long iterationDuration =std::chrono::duration_cast<std::chrono::milliseconds>(clockIter2 - clockIter1).count();
      timeTotal += iterationDuration;

      long currentGlobalIteration = iter + 1;
      double progress = (double)currentGlobalIteration / (maxIteration + 1);
      double avgMillisPerIter = (double)timeTotal / currentGlobalIteration;
      long remainingIterations = (maxIteration + 1) - currentGlobalIteration;
      long etaMillis = (long)(avgMillisPerIter * remainingIterations);

      int etaSeconds = etaMillis / 1000;
      int etaMinutes = etaSeconds / 60;
      int etaRemSeconds = etaSeconds % 60;

      // ===== SPEED =====
      double totalKeysTested =(double)currentGlobalIteration * itercount;
      double speedKeysPerSec =totalKeysTested / (timeTotal / 1000.0);
      double speedMKeys = speedKeysPerSec / 1000000.0;

      printf("\r");
      printProgressBar(progress);
      printf(" %6.2f%% |%8.2f Mkeys/s | ETA: %02dm %02ds | iteration %d/%lld", progress * 100.0, speedMKeys, etaMinutes, etaRemSeconds, iter, maxIteration);
      fflush(stdout);

   }
   auto clockEnd = std::chrono::high_resolution_clock::now();
   long long timeTotal2 =std::chrono::duration_cast<std::chrono::milliseconds>(clockEnd - clockStart).count();

   long long totalKeysTested =(long long)(maxIteration) * itercount;
   double totalSeconds = timeTotal2 / 1000.0;
   double avgKeysPerSec =totalKeysTested / totalSeconds;
   double avgMKeysPerSec =avgKeysPerSec / 1000000.0;

   printf("Tempo total: %.2f segundos\n", totalSeconds);
   printf("Total de chaves testadas: %lld\n", totalKeysTested);
   printf("Velocidade média: %.2f Mkeys/s\n", avgMKeysPerSec);
}


void startSecp256k1ModeWIFScanL(uint8_t * gTableXCPU, uint8_t * gTableYCPU, uint64_t * inputHashBufferCPU, int positions, const char* WIFprivKeyCPU)
{
   printf("Mode Linear WIF scan\n");    
   printf("Chave Parcial char: %s\n", WIFprivKeyCPU);

   int posicoesCPU[52];
   int totalPosicoesCPUtemp;


   GPUSecp *gpuSecp = new GPUSecp(
      gTableXCPU,
      gTableYCPU,
      inputHashBufferCPU,
      WIFprivKeyCPU,
      posicoesCPU,
      totalPosicoesCPUtemp
   );

   long timeTotal = 0;

   totalPosicoesCPUtemp = positions;
   for(int x=0;x<totalPosicoesCPUtemp;x++){
      posicoesCPU[x]=x;
   };

   long long possibilidades = pow(58, totalPosicoesCPUtemp) - 1;

   int itercount = COUNT_CUDA_THREADS * THREAD_MULT;
   long long maxIteration = (possibilidades + itercount - 1) / itercount;

   int maxscan = 52 - totalPosicoesCPUtemp + 1;

   long totalGlobalIterations = (long)maxscan * (maxIteration + 1);

   printf("Total iterações globais: %ld\n", totalGlobalIterations);
   auto clockStart = std::chrono::high_resolution_clock::now();

   for(int ms = 0; ms < maxscan; ms++){
      gpuSecp->updatemodohex(posicoesCPU, totalPosicoesCPUtemp);

      for(int iter = 0; iter < maxIteration + 1; iter++){
         const auto clockIter1 = std::chrono::high_resolution_clock::now();

         
         gpuSecp->doIterationSecp256k1Books(iter);
         gpuSecp->doPrintOutput();

         const auto clockIter2 = std::chrono::high_resolution_clock::now();
         long iterationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(clockIter2 - clockIter1).count();

         timeTotal += iterationDuration;

            // ===== CÁLCULO GLOBAL =====

         long currentGlobalIteration = (long)ms * (maxIteration + 1) + iter + 1;
         double progress = (double)currentGlobalIteration / totalGlobalIterations;
         long elapsedMillis = timeTotal;

         double avgMillisPerIter = (double)elapsedMillis / currentGlobalIteration;
         long remainingIterations = totalGlobalIterations - currentGlobalIteration;
         long etaMillis = (long)(avgMillisPerIter * remainingIterations);
         int etaSeconds = etaMillis / 1000;
         int etaMinutes = etaSeconds / 60;
         int etaRemSeconds = etaSeconds % 60;

            // ===== SPEED =====

         double totalKeysTested = (double)currentGlobalIteration * itercount;
         double speedKeysPerSec = totalKeysTested / (elapsedMillis / 1000.0);
         double speedMKeys = speedKeysPerSec / 1000000.0;

         printf("\r");
         printProgressBar(progress); 
         printf(" %6.2f%% | %8.2f Mkeys/s | ETA: %02dm %02ds | Scan %d/%d", progress * 100.0, speedMKeys, etaMinutes, etaRemSeconds, ms + 1, maxscan);
         printf("\n");
         printStrCPU(WIFprivKeyCPU, posicoesCPU, totalPosicoesCPUtemp);
         printf("\033[1A");
         fflush(stdout);
      }
      for(int r = 0; r < totalPosicoesCPUtemp; r++) posicoesCPU[r]++;
   }

   auto clockEnd = std::chrono::high_resolution_clock::now();
   long long timeTotal2 =std::chrono::duration_cast<std::chrono::milliseconds>(clockEnd - clockStart).count();

   long long totalKeysTested =(long long)(totalGlobalIterations*itercount);
   double totalSeconds = timeTotal2 / 1000.0;
   double avgKeysPerSec =totalKeysTested / totalSeconds;
   double avgMKeysPerSec =avgKeysPerSec / 1000000.0;

   printf("\nTempo total: %.2f segundos\n", totalSeconds);
   printf("Total de chaves testadas: %lld\n", totalKeysTested);
   printf("Velocidade média: %.2f Mkeys/s\n", avgMKeysPerSec);
}

unsigned long long calcularCombinacoes(int n, int k){
   if (k < 0 || k > n) return 0;
   if (k == 0 || k == n) return 1;

   if (k > n - k)
      k = n - k; // otimização

   unsigned long long resultado = 1;

   for (int i = 0; i < k; i++){
      resultado = resultado * (n - i);
      resultado = resultado / (i + 1);
   }
   return resultado;
}

int proximaCombinacao(int *posicoesCPU, int k, int max){
   for(int i = k - 1; i >= 0; i--){
      if(posicoesCPU[i] < max - k + i){
         posicoesCPU[i]++;

         for(int j = i + 1; j < k; j++)
            posicoesCPU[j] = posicoesCPU[j - 1] + 1;
            return 1;
      }
   }
   return 0;
}

void startSecp256k1ModeWIFComb(uint8_t * gTableXCPU, uint8_t * gTableYCPU, uint64_t * inputHashBufferCPU, int positions, const char* WIFprivKeyCPU){
   printf("Mode Combination WIF scan\n");
       
   int posicoesCPU[52] = {0};
   int totalPosicoesCPUtemp;

   GPUSecp *gpuSecp = new GPUSecp(
      gTableXCPU,
      gTableYCPU,
      inputHashBufferCPU,
      WIFprivKeyCPU,
      posicoesCPU,
      totalPosicoesCPUtemp
   );

   long timeTotal = 0;
   //inicializa o array posicoes
   totalPosicoesCPUtemp = positions; 
   for(int x=0;x<totalPosicoesCPUtemp;x++){
      posicoesCPU[x]=x;
   };

   long long possibilidades = 1;
   for(int i = 0; i < totalPosicoesCPUtemp; i++)possibilidades *= 58;

   printf("Chave Parcial char: %s\n", WIFprivKeyCPU);
   printf("Possibilidades por combinação (58^%d): %llu\n", totalPosicoesCPUtemp, possibilidades);

   int itercount = COUNT_CUDA_THREADS * THREAD_MULT;
   long long maxIteration = (possibilidades + itercount - 1) / itercount;

    

   unsigned long long maxcombination = calcularCombinacoes(52, totalPosicoesCPUtemp);
   long totalGlobalIterations = (long)maxcombination * (maxIteration + 1);

   printf("Numedo de combinações %lld Total iterações globais: %ld\n",maxcombination ,totalGlobalIterations);
   printf("cada iteração calcula %d chaves resultando no maximo de %lld iterações em cada combinação.",itercount ,maxIteration);

   auto clockStart = std::chrono::high_resolution_clock::now();
   for (int ms = 0; ms < maxcombination; ms++){
         for (int iter = 0; iter < maxIteration + 1; iter++){
            const auto clockIter1 = std::chrono::high_resolution_clock::now();

            gpuSecp->updatemodohex(posicoesCPU, totalPosicoesCPUtemp);
            gpuSecp->doIterationSecp256k1Books(iter);
            gpuSecp->doPrintOutput();

            const auto clockIter2 = std::chrono::high_resolution_clock::now();
            long iterationDuration =std::chrono::duration_cast<std::chrono::milliseconds>(clockIter2 - clockIter1).count();
            timeTotal += iterationDuration;

            // ===== CÁLCULO GLOBAL =====
            long currentGlobalIteration =(long)ms * (maxIteration + 1) + iter + 1;
            double progress = (double)currentGlobalIteration / totalGlobalIterations;
            long elapsedMillis = timeTotal;
            double avgMillisPerIter = (double)elapsedMillis / currentGlobalIteration;
            long remainingIterations = totalGlobalIterations - currentGlobalIteration;
            long etaMillis = (long)(avgMillisPerIter * remainingIterations);
            int etaSeconds = etaMillis / 1000;
            int etaMinutes = etaSeconds / 60;
            int etaRemSeconds = etaSeconds % 60;

            // ===== SPEED =====
            double totalKeysTested = (double)currentGlobalIteration * itercount;
            double speedKeysPerSec = totalKeysTested / (elapsedMillis / 1000.0);
            double speedMKeys = speedKeysPerSec / 1000000.0;
            printf("\r");
            printProgressBar(progress);
            printf(" %6.2f%% | %8.2f Mkeys/s | ETA: %02dm %02ds | comb %d/%lld | iter %d/%lld", progress * 100.0, speedMKeys, etaMinutes, etaRemSeconds, ms + 1, maxcombination, iter, maxIteration);
            printf("\n");
            printStrCPU(WIFprivKeyCPU, posicoesCPU, totalPosicoesCPUtemp);
            printf("\033[1A");
            fflush(stdout);
      }
      proximaCombinacao(posicoesCPU,totalPosicoesCPUtemp, 52);
   }
   auto clockEnd = std::chrono::high_resolution_clock::now();
   long long timeTotal2 =std::chrono::duration_cast<std::chrono::milliseconds>(clockEnd - clockStart).count();

   long long totalKeysTested =(long long)(totalGlobalIterations*itercount);
   double totalSeconds = timeTotal2 / 1000.0;
   double avgKeysPerSec =totalKeysTested / totalSeconds;
   double avgMKeysPerSec =avgKeysPerSec / 1000000.0;

   printf("\nTempo total: %.2f segundos\n", totalSeconds);
   printf("Total de chaves testadas: %lld\n", totalKeysTested);
   printf("Velocidade média: %.2f Mkeys/s\n", avgMKeysPerSec);
}


void increaseStackSizeCPU() {
   const rlim_t cpuStackSize = SIZE_CPU_STACK;
   struct rlimit rl;
   int result;

   printf("Increasing Stack Size to %lu \n", cpuStackSize);

   result = getrlimit(RLIMIT_STACK, &rl);
   if (result == 0)
   {
      if (rl.rlim_cur < cpuStackSize)
      {
         rl.rlim_cur = cpuStackSize;
         result = setrlimit(RLIMIT_STACK, &rl);
         if (result != 0)
         {
            fprintf(stderr, "setrlimit returned result = %d\n", result);
         }
      }
   }
}

int main(int argc, char **argv) {
   printf("iniciando a cambuca \n");

    bool modeWIFX   = false;
    bool modeLinear = false;
    bool modeComb   = false;
    int positions = -1;   // numero de posicoes para modo linear e comb
    bool key = false;
    char strCPUload[53] = {0};// 53 \0
    bool showHelp = false;
    bool compressed = false;

    const struct option long_options[] = {
        {"linear",      no_argument,       nullptr, 'l'},
        {"combination", no_argument,       nullptr, 'c'},
        {"WIFx",        no_argument,       nullptr, 'x'},
        {"positions",   required_argument, nullptr, 'p'},
        {"key",         required_argument, nullptr, 'k'},
        {"help",        no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}   // <--- precisa disso so nao sei o pq 
    };

    int opt;

     while ((opt = getopt_long(argc, argv, "lcxp:k:h", long_options, nullptr)) != -1){ //:significa que recebe argumentos seu bobao
        switch (opt)
        {
            case 'l':
                modeLinear = true;
                break;

            case 'c':
                modeComb = true;
                break;

            case 'x':
                modeWIFX = true;
                break;

            case 'p':
                positions = atoi(optarg);
                if (positions <= 0 || positions > 8)
                {
                    fprintf(stderr, "Erro: --calma ai meu chapa\n");
                    return 1;
                }
                break;

            case 'h':
                showHelp = true;
                break;

            case 'k':
            {
               size_t len = strlen(optarg);

               if (len == 51) {
                  compressed = false;
               }
               else if (len == 52) {
                  compressed = true;
               }
               else {
               fprintf(stderr, "Erro: WIF deve ter 51 ou 52 caracteres\n");
               return 1;
            }

            strcpy(strCPUload, optarg);

            key = true;
            break;
}

            default:
                showHelp = true;
                break;
        }
    }

    int modeCount = 0;
    if (modeLinear) modeCount++;
    if (modeComb)   modeCount++;
    if (modeWIFX)   modeCount++;

    if (modeCount == 0)
    {
        showHelp=true;
    }

    if (modeCount > 1)
    {
        fprintf(stderr, "Erro: selecione um modo por vez\n");
        return 1;
    }

    if (positions != -1 && modeWIFX)
    {
       fprintf(stderr, "Erro: -p/--positions não pode ser usado com -x\n");
       return 1;
    }

    if (positions == -1){
       if (modeLinear)positions = 5;
        if (modeComb)positions = 3;
    }


        if (!key){
            if (!lerChaveDeArquivo("chave.txt", strCPUload))
            return 1;
        }

    if (modeWIFX)
        if (!validarChaveWIF(strCPUload, true)) // true permite x
            return 1;
        
    if (modeLinear || modeComb)
        if (!validarChaveWIF(strCPUload, false)) //nao permite x
            return 1;



   if (showHelp){

    printf("\nUsage: %s [MODE] [OPTIONS]\n\n", argv[0]);

    printf("Modes:\n");
    printf("  -x, --WIFx        Scattered WIF mode (use '0' for unknown chars)\n");
    printf("  -l, --linear      Sequential scan of missing WIF characters\n");
    printf("  -c, --combination Test all combinations of unknown positions\n\n");

    printf("Options:\n");
    printf("  -p N              Unknown WIF positions (-c, -l modes only)\n");
    printf("  -k WIF            Partial key (otherwise reads 'chave.txt')\n");
    printf("  -h                Show this help message\n\n");

    printf("Example:\n");
    printf("  %s -x -k L4rK1yDtCWekvXuE6oXD9jCYfFNV2cWRp0uPLBc0U2z0Tri0oyY0\n", argv[0]);
    printf("  %s -l -p 5 -k L4rK1yDtCWekvXuE6oXD9jCYfFNV2cWRpVuPLBcCU2z8Trixxxxx\n", argv[0]);
    printf("  %s -c -p 3 -k L4rKxyDtCWekvXuE6oXD9jCYfFNVxcWRpVuPLBcCU2z8TrisoyYx\n\n", argv[0]);


    printf("Donations: bc1qs850jrz5ktl5vwpma0sz40z29392wrzx9cevze\n");
    return 0;
   }


   increaseStackSizeCPU();
   mergeHashes(NAME_HASH_FOLDER, NAME_HASH_BUFFER);

   uint8_t* gTableXCPU = new uint8_t[COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT];
   uint8_t* gTableYCPU = new uint8_t[COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT];

   loadGTable(gTableXCPU, gTableYCPU);

   uint64_t* inputHashBufferCPU = new uint64_t[COUNT_INPUT_HASH];

   loadInputHash(inputHashBufferCPU);


    if (modeWIFX)
        startSecp256k1ModeWIFX(gTableXCPU, gTableYCPU, inputHashBufferCPU, strCPUload);
    if (modeLinear)
        startSecp256k1ModeWIFScanL(gTableXCPU, gTableYCPU, inputHashBufferCPU, positions, strCPUload);
     
    if (modeComb)
        startSecp256k1ModeWIFComb(gTableXCPU, gTableYCPU, inputHashBufferCPU, positions, strCPUload);
    

   delete[] gTableXCPU;
   delete[] gTableYCPU;
   delete[] inputHashBufferCPU;

   printf("Finish \n");
   return 0;
}
