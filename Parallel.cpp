#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <time.h>
#include <iostream>
#include <sstream> //this header file is needed when using stringstream
#include <fstream>
#include <string>
#include <pthread.h>
#include <string.h>
#include <semaphore.h>


#define MNIST_TESTING_SET_IMAGE_FILE_NAME "data/t10k-images-idx3-ubyte"  ///< MNIST image testing file in the data folder
#define MNIST_TESTING_SET_LABEL_FILE_NAME "data/t10k-labels-idx1-ubyte"  ///< MNIST label testing file in the data folder

#define HIDDEN_WEIGHTS_FILE "net_params/hidden_weights.txt"
#define HIDDEN_BIASES_FILE "net_params/hidden_biases.txt"
#define OUTPUT_WEIGHTS_FILE "net_params/out_weights.txt"
#define OUTPUT_BIASES_FILE "net_params/out_biases.txt"

#define NUMBER_OF_INPUT_CELLS 784   ///< use 28*28 input cells (= number of pixels per MNIST image)
#define NUMBER_OF_HIDDEN_CELLS 256   ///< use 256 hidden cells in one hidden layer
#define NUMBER_OF_OUTPUT_CELLS 10   ///< use 10 output cells to model 10 digits (0-9)

#define MNIST_MAX_TESTING_IMAGES 10000                      ///< number of images+labels in the TEST file/s
#define MNIST_IMG_WIDTH 28                                  ///< image width in pixel
#define MNIST_IMG_HEIGHT 28                                 ///< image height in pixel

#define INPUT_THREADS_NUM 1
#define OUTPUT_THREADS_NUM 10

using namespace std;

struct Hidden_Node{
    double weights[28*28];
    double bias;
    double output;
};
struct Output_Node{
    double weights[256];
    double bias;
    double output;
};
int hidden_threads_num;

typedef struct MNIST_ImageFileHeader MNIST_ImageFileHeader;
typedef struct MNIST_LabelFileHeader MNIST_LabelFileHeader;

typedef struct MNIST_Image MNIST_Image;
typedef uint8_t MNIST_Label;
typedef struct Hidden_Node Hidden_Node;
typedef struct Output_Node Output_Node;
Hidden_Node hidden_nodes[NUMBER_OF_HIDDEN_CELLS];
Output_Node output_nodes[NUMBER_OF_OUTPUT_CELLS];

pthread_t* hiddenThreads;
pthread_t inputThread;
pthread_t outputThreads[OUTPUT_THREADS_NUM];
pthread_t resultThread;
bool writingInputs = false;

 // open MNIST files
FILE *imageFile, *labelFile;

//Semaphors : 
typedef struct {
    sem_t occupied;
    sem_t empty;
} inputSemaphors_t;
inputSemaphors_t inputSemaphors;

typedef struct {
    sem_t *empty;
} hiddenSemaphors_t;
hiddenSemaphors_t hiddenSemaphors;

typedef struct {
    sem_t occupied[OUTPUT_THREADS_NUM];
} outputSemaphors_t;
outputSemaphors_t outputSemaphors;

typedef struct {
    sem_t occupied;
    sem_t empty;
} resultSemaphors_t;
resultSemaphors_t resultSemaphors;

typedef struct {
    sem_t occupied;
    sem_t empty;
} consoleSemaphore_t;
consoleSemaphore_t consoleSemaphore;

void consoleSemaphoreInit()
{
    sem_init(&consoleSemaphore.empty, 0, 1);
    sem_init(&consoleSemaphore.occupied, 0, 1);
}

void resultSemaphorsInit()
{
    // for(int i = 0 ; i < OUTPUT_THREADS_NUM; i++)
    // {
        sem_init(&resultSemaphors.occupied, 0, 0);
        sem_init(&resultSemaphors.empty, 0, OUTPUT_THREADS_NUM);
    // }
}

void outputSemaphorsInit()
{
    for(int i = 0 ; i < OUTPUT_THREADS_NUM ; i++)
    {
        sem_init(&outputSemaphors.occupied[i], 0, 0);
        // sem_init(&outputSemaphors.empty[i], 0, 1);
    }
}
void hiddenSemaphorsInit()
{
    hiddenSemaphors.empty = (sem_t*) malloc(sizeof(sem_t) * hidden_threads_num);
    for(int i = 0 ; i < hidden_threads_num; i++)
    {
        // sem_init(&hiddenSemaphors.occupied[i], 0, 0);
        sem_init(&hiddenSemaphors.empty[i], 0, OUTPUT_THREADS_NUM);
    }
    
}

void inputSemaphorsInit()
{
    // inputSemaphors.occupied = (sem_t *) malloc(sizeof(sem_t) * hidden_threads_num);
    // inputSemaphors.empty    = (sem_t *) malloc(sizeof(sem_t) * hidden_threads_num);
    // for(int i = 0 ; i < hidden_threads_num; i++)
    // { 
        sem_init(&inputSemaphors.occupied, 0,0);
        sem_init(&inputSemaphors.empty,0, hidden_threads_num);
    // }
}
/**
 * @brief Data block defining a hidden cell
 */


/**
 * @brief Data block defining an output cell
 */


/**
 * @brief Data block defining a MNIST image
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_Image{
    uint8_t pixel[28*28];
};

/**
 * @brief Data block defining a MNIST image file header
 * @attention The fields in this structure are not used.
 * What matters is their byte size to move the file pointer
 * to the first image.
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_ImageFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
};

/**
 * @brief Data block defining a MNIST label file header
 * @attention The fields in this structure are not used.
 * What matters is their byte size to move the file pointer
 * to the first label.
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_LabelFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
};

/**
 * @details Set cursor position to given coordinates in the terminal window
 */


//Shared variables between threads :
MNIST_Image* img;
MNIST_Image* getImg;
MNIST_Label lbl;


void locateCursor(const int row, const int col){
    printf("%c[%d;%dH",27,row,col);
}

/**
 * @details Clear terminal screen by printing an escape sequence
 */

void clearScreen(){
    printf("\e[1;1H\e[2J");
}

/**
 * @details Outputs a 28x28 MNIST image as charachters ("."s and "X"s)
 */

void displayImage(MNIST_Image *img, int row, int col){

    char imgStr[(MNIST_IMG_HEIGHT * MNIST_IMG_WIDTH)+((col+1)*MNIST_IMG_HEIGHT)+1];
    strcpy(imgStr, "");

    for (int y=0; y<MNIST_IMG_HEIGHT; y++){

        for (int o=0; o<col-2; o++) strcat(imgStr," ");
        strcat(imgStr,"|");

        for (int x=0; x<MNIST_IMG_WIDTH; x++){
            strcat(imgStr, img->pixel[y*MNIST_IMG_HEIGHT+x] ? "X" : "." );
        }
        strcat(imgStr,"\n");
    }

    if (col!=0 && row!=0) locateCursor(row, 0);
    printf("%s",imgStr);
}

/**
 * @details Outputs a 28x28 text frame at a defined screen position
 */

void displayImageFrame(int row, int col){

    if (col!=0 && row!=0) locateCursor(row, col);

    printf("------------------------------\n");

    for (int i=0; i<MNIST_IMG_HEIGHT; i++){
        for (int o=0; o<col-1; o++) printf(" ");
        printf("|                            |\n");
    }

    for (int o=0; o<col-1; o++) printf(" ");
    printf("------------------------------");

}

/**
 * @details Outputs reading progress while processing MNIST testing images
 */

void displayLoadingProgressTesting(int imgCount, int y, int x){

    float progress = (float)(imgCount+1)/(float)(MNIST_MAX_TESTING_IMAGES)*100;

    if (x!=0 && y!=0) locateCursor(y, x);

    printf("Testing image No. %5d of %5d images [%d%%]\n                                  ",(imgCount+1),MNIST_MAX_TESTING_IMAGES,(int)progress);

}

/**
 * @details Outputs image recognition progress and error count
 */

void displayProgress(int imgCount, int errCount, int y, int x){

    double successRate = 1 - ((double)errCount/(double)(imgCount+1));

    if (x!=0 && y!=0) locateCursor(y, x);

    printf("Result: Correct=%5d  Incorrect=%5d  Success-Rate= %5.2f%% \n",imgCount+1-errCount, errCount, successRate*100);


}

/**
 * @details Reverse byte order in 32bit numbers
 * MNIST files contain all numbers in reversed byte order,
 * and hence must be reversed before using
 */

uint32_t flipBytes(uint32_t n){

    uint32_t b0,b1,b2,b3;

    b0 = (n & 0x000000ff) <<  24u;
    b1 = (n & 0x0000ff00) <<   8u;
    b2 = (n & 0x00ff0000) >>   8u;
    b3 = (n & 0xff000000) >>  24u;

    return (b0 | b1 | b2 | b3);

}

/**
 * @details Read MNIST image file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

void readImageFileHeader(FILE *imageFile, MNIST_ImageFileHeader *ifh){

    ifh->magicNumber =0;
    ifh->maxImages   =0;
    ifh->imgWidth    =0;
    ifh->imgHeight   =0;

    fread(&ifh->magicNumber, 4, 1, imageFile);
    ifh->magicNumber = flipBytes(ifh->magicNumber);

    fread(&ifh->maxImages, 4, 1, imageFile);
    ifh->maxImages = flipBytes(ifh->maxImages);

    fread(&ifh->imgWidth, 4, 1, imageFile);
    ifh->imgWidth = flipBytes(ifh->imgWidth);

    fread(&ifh->imgHeight, 4, 1, imageFile);
    ifh->imgHeight = flipBytes(ifh->imgHeight);
}

/**
 * @details Read MNIST label file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

void readLabelFileHeader(FILE *imageFile, MNIST_LabelFileHeader *lfh){

    lfh->magicNumber =0;
    lfh->maxImages   =0;

    fread(&lfh->magicNumber, 4, 1, imageFile);
    lfh->magicNumber = flipBytes(lfh->magicNumber);

    fread(&lfh->maxImages, 4, 1, imageFile);
    lfh->maxImages = flipBytes(lfh->maxImages);

}

/**
 * @details Open MNIST image file and read header info
 * by reading the header info, the read pointer
 * is moved to the position of the 1st IMAGE
 */

FILE *openMNISTImageFile(char *fileName){

    FILE *imageFile;
    imageFile = fopen (fileName, "rb");
    if (imageFile == NULL) {
        printf("Abort! Could not fine MNIST IMAGE file: %s\n",fileName);
        exit(0);
    }

    MNIST_ImageFileHeader imageFileHeader;
    readImageFileHeader(imageFile, &imageFileHeader);

    return imageFile;
}


/**
 * @details Open MNIST label file and read header info
 * by reading the header info, the read pointer
 * is moved to the position of the 1st LABEL
 */

FILE *openMNISTLabelFile(char *fileName){

    FILE *labelFile;
    labelFile = fopen (fileName, "rb");
    if (labelFile == NULL) {
        printf("Abort! Could not find MNIST LABEL file: %s\n",fileName);
        exit(0);
    }

    MNIST_LabelFileHeader labelFileHeader;
    readLabelFileHeader(labelFile, &labelFileHeader);

    return labelFile;
}

/**
 * @details Returns the next image in the given MNIST image file
 */

void getImage(FILE *imageFile){

    // MNIST_Image img;
    size_t result;
    result = fread(getImg, sizeof(MNIST_Image), 1, imageFile);
    if (result!=1) {
        printf("\nError when reading IMAGE file! Abort!\n");
        exit(1);
    }

    // return img;
}

/**
 * @details Returns the next label in the given MNIST label file
 */

MNIST_Label getLabel(FILE *labelFile){

    MNIST_Label lbl;
    size_t result;
    result = fread(&lbl, sizeof(lbl), 1, labelFile);
    if (result!=1) {
        printf("\nError when reading LABEL file! Abort!\n");
        exit(1);
    }

    return lbl;
}

/**
 * @brief allocate weights and bias to respective hidden cells
 */

void allocateHiddenParameters(){
    int idx = 0;
    int bidx = 0;
    ifstream weights(HIDDEN_WEIGHTS_FILE);
    for(string line; getline(weights, line); )   //read stream line by line
    {
        stringstream in(line);
        for (int i = 0; i < 28*28; ++i){
            in >> hidden_nodes[idx].weights[i];
      }
      idx++;
    }
    weights.close();

    ifstream biases(OUTPUT_BIASES_FILE);
    for(string line; getline(biases, line); )   //read stream line by line
    {
        stringstream in(line);
        in >> hidden_nodes[bidx].bias;
        bidx++;
    }
    biases.close();

}

/**
 * @brief allocate weights and bias to respective output cells
 */

void allocateOutputParameters(){
    int idx = 0;
    int bidx = 0;
    ifstream weights(OUTPUT_WEIGHTS_FILE); //"layersinfo.txt"
    for(string line; getline(weights, line); )   //read stream line by line
    {
        stringstream in(line);
        for (int i = 0; i < 256; ++i){
            in >> output_nodes[idx].weights[i];
      }
      idx++;
    }
    weights.close();

    ifstream biases(OUTPUT_BIASES_FILE);
    for(string line; getline(biases, line); )   //read stream line by line
    {
        stringstream in(line);
        in >> output_nodes[bidx].bias;
        bidx++;
    }
    biases.close();

}

/**
 * @details The output prediction is derived by finding the maxmimum output value
 * and returning its index (=0-9 number) as the prediction.
 */

int getNNPrediction(){

    double maxOut = 0;
    int maxInd = 0;

    for (int i=0; i<NUMBER_OF_OUTPUT_CELLS; i++){

        if (output_nodes[i].output > maxOut){
            maxOut = output_nodes[i].output;
            maxInd = i;
        }
    }

    return maxInd;

}

/**
 * @details test the neural networks to obtain its accuracy when classifying
 * 10k images.
 */
void swapGetAndOutImage()
{
    MNIST_Image* temp;
    temp = getImg;
    getImg = img;
    img = temp;
}
void *getInput(void* j)
{
    imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME); 
    // screen output for monitoring progress
    displayImageFrame(7,5);
    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++)
    {
       
        
        // Reading next image 
        getImage(imageFile);

        //lock
        for(int i = 0; i < hidden_threads_num; i++)
            sem_wait(&inputSemaphors.empty);
        
        swapGetAndOutImage();

        sem_wait(&consoleSemaphore.empty);
        displayLoadingProgressTesting(imgCount,5,5);
        displayImage(img, 8,6);
        sem_post(&consoleSemaphore.occupied);

        //unlock
        for(int i = 0; i < hidden_threads_num; i++)
            sem_post(&inputSemaphors.occupied);
       
        
    }
    pthread_exit(NULL);
}
void *calcHiddenCells(void* param)
{
    int* threadNum = (int*) param;
    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++)
    {
        
       
        // sem_wait(&hiddenSemaphors.empty[*threadNum]);

        for(int i = 0 ; i < OUTPUT_THREADS_NUM; i++)
            sem_wait(&hiddenSemaphors.empty[*threadNum]);
        sem_wait(&inputSemaphors.occupied);
        // cerr << "T2" << endl;
        for(int j = (*threadNum) * (NUMBER_OF_HIDDEN_CELLS / hidden_threads_num);
            j < ((*threadNum) + 1) * (NUMBER_OF_HIDDEN_CELLS / hidden_threads_num);
            j++)
        {
            hidden_nodes[j].output = 0;
            for (int z = 0; z < NUMBER_OF_INPUT_CELLS; z++) {
                hidden_nodes[j].output += img->pixel[z] * hidden_nodes[j].weights[z];
            }
            hidden_nodes[j].output += hidden_nodes[j].bias;
            hidden_nodes[j].output = (hidden_nodes[j].output >= 0) ?  hidden_nodes[j].output : 0;
        }
        // cerr << "T2\n";
        for(int i = 0 ; i < OUTPUT_THREADS_NUM; i++)
            sem_post(&outputSemaphors.occupied[i]);
    
        // sem_post(&hiddenSemaphors.occupied[*threadNum]);
        sem_post(&inputSemaphors.empty);
    }
    pthread_exit(NULL);
}
void *calcOutputCells(void* param)
{
    int* threadNum = (int*) param;
    for(int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++)
    {
       

        for(int i = 0; i < hidden_threads_num; i++)
            sem_wait(&outputSemaphors.occupied[*threadNum]);
        sem_wait(&resultSemaphors.empty);
        int i = (*threadNum);
        output_nodes[i].output = 0;
        
        for (int j = 0; j < NUMBER_OF_HIDDEN_CELLS; j++) {
            output_nodes[i].output += hidden_nodes[j].output * output_nodes[i].weights[j];
        }
        output_nodes[i].output = 1/(1+ exp(-1* (output_nodes[i].output + output_nodes[i].bias)));
        
        sem_post(&resultSemaphors.occupied);
        for(int i = 0; i < hidden_threads_num; i++)
            sem_post(&hiddenSemaphors.empty[i]);
        
        

    }

    pthread_exit(NULL);
}

void *showResults(void* param)
{
    // number of incorrect predictions
    int errCount = 0;
        // Loop through all images in the file
    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++)
    {
        for(int i = 0; i < OUTPUT_THREADS_NUM; i++)
            sem_wait(&resultSemaphors.occupied);

        lbl = getLabel(labelFile);
        int predictedNum = getNNPrediction();
        if (predictedNum!=lbl) 
            errCount++;

        sem_wait(&consoleSemaphore.occupied);
        printf("\n      Prediction: %d   Actual: %d ",predictedNum, lbl);
        displayProgress(imgCount, errCount, 5, 66);
        sem_post(&consoleSemaphore.empty);

        for(int i = 0; i < OUTPUT_THREADS_NUM; i++)
            sem_post(&resultSemaphors.empty);

       
    }
    // Close files
    fclose(imageFile);
    fclose(labelFile);
    
    // pthread_exit(NULL);
}

void testNN(){

    //Initializing Semaphors : 
    inputSemaphorsInit();
    hiddenSemaphorsInit();
    resultSemaphorsInit();
    consoleSemaphoreInit();
    //Creating threads :
    int inputParam = 0;
    pthread_create(&inputThread, NULL, getInput, (void*)&inputParam);

    hiddenThreads = (pthread_t *) malloc(sizeof(pthread_t) * hidden_threads_num);
    int *hiddenThreadsParam = (int *) malloc(sizeof(int) * hidden_threads_num);
    for(int j = 0; j < hidden_threads_num ; j++)
    {
        hiddenThreadsParam[j] = j;
        pthread_create(&hiddenThreads[j], NULL, calcHiddenCells, (void*)(hiddenThreadsParam + j));
    }

    int outputThreadsParam[OUTPUT_THREADS_NUM];
    for(int j = 0; j < OUTPUT_THREADS_NUM ; j++)
    {
        outputThreadsParam[j] = j;
        pthread_create(&outputThreads[j], NULL, calcOutputCells, (void*)(outputThreadsParam + j));
    }

    int resultParam = 0;
    pthread_create(&resultThread, NULL, showResults,(void*)&resultParam);
    // showResults((void*)&resultParam);


    //Waitign for all Threads
    pthread_join(inputThread, NULL);
    for(int i = 0; i < hidden_threads_num; i++)
        pthread_join(hiddenThreads[i], NULL);
    for(int i = 0; i < OUTPUT_THREADS_NUM; i++)
        pthread_join(outputThreads[i], NULL);
    pthread_join(resultThread, NULL);

}

void mallocDataStructures()
{
    img = (MNIST_Image*) malloc(sizeof(MNIST_Image) * 1);
    getImg = (MNIST_Image*) malloc(sizeof(MNIST_Image) * 1);
}
int main(int argc, const char * argv[]) {

    mallocDataStructures();

    cout << "Please enter number of hidden threads" << endl;
    cin >> hidden_threads_num;
    // remember the time in order to calculate processing time at the end
    time_t startTime = time(NULL);

    // clear screen of terminal window
    clearScreen();
    printf("    MNIST-NN: a simple 2-layer neural network processing the MNIST handwriting images\n");

    // alocating respective parameters to hidden and output layer cells
    allocateHiddenParameters();
    allocateOutputParameters();

    //test the neural network
    testNN();

    locateCursor(38, 5);

    // calculate and print the program's total execution time
    time_t endTime = time(NULL);
    double executionTime = difftime(endTime, startTime);
    printf("\n\n    DONE! Total execution time: %.1f sec\n\n",executionTime);

    return 0;
}
