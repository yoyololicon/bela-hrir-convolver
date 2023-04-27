/*
 ____  _____ _        _    
| __ )| ____| |      / \   
|  _ \|  _| | |     / _ \  
| |_) | |___| |___ / ___ \ 
|____/|_____|_____/_/   \_\

http://bela.io

C++ Real-Time Audio Programming with Bela - Lecture 5: Classes and Objects
additive-synth: an example implementing an additive synthesiser based on an
                array of Wavetable oscillator objects
*/

#include <Bela.h>
#include <libraries/Fft/Fft.h>
#include <libraries/Gui/Gui.h>
#include <libraries/GuiController/GuiController.h>
#include <libraries/Scope/Scope.h>
#include <libraries/AudioFile/AudioFile.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <sstream>
#include <fstream>

#include "KDTree.h"


// Constants that define the program behaviour
const std::string kInfoFile = "info.txt";
const std::string kSimplexFile = "simplices.txt";
const unsigned int kHopSize = 32;
const unsigned int kBufferSize = 2048;
const unsigned int kNeighbours = 24;

// Circular buffer and pointer for assembling a window of samples
std::vector<float> gOutputBuffer[2];
int gOutputBufferPointer = 0;
int gHopCounter = 0;

// HRIR directory
std::string gHRIRDir = "KU100_NF100";

// Name of the sound file (in project folder)
std::string gSoundFilename = "drum-loop.wav"; 

// Buffers to hold the audio
std::vector<float> gSoundBuffer;

// Read pointer into the audio buffer
unsigned int gReadPointer = 0;

// Buffers to hold HRIRs
std::vector<std::vector<float>> gHRIRInterleaved;
std::vector<std::vector<float>> gHRIRCoordinates;
std::vector<std::vector<float>> gSimplicesCentres;
std::vector<std::vector<int>> gNeighborSimplices;
std::vector<std::vector<int>> gSimplices;
std::vector<std::vector<float>> gInverseMatrices;
std::vector<float> gSampledHRIR[2];
std::vector<float> gCurrentHRIR[2];
std::vector<float> gHRIRInterpolationGrad[2];

int gErrorCounts = 0;

// KD tree for sample retrieval
KDTree gCoordTree;

// Coordinates of the sound source
float gX, gY, gZ;

// HRIR specifications
int gHRIRSamples = 100;
float gRadius = 0;



// Browser-based GUI to adjust parameters
Gui gGui;
GuiController gGuiController;

// Browser-based oscilloscope
Scope gScope;

// Thread for HRIR processing
AuxiliaryTask gSampleRetrieveTask;

// Cached variables for the second thread
float gCachedX, gCachedY, gCachedZ;


// This function runs in an auxiliary task on Bela, calling process_fft
void weighted_hrir_background(void *)
{
	std::vector<float> point {gCachedX, gCachedY, gCachedZ};
	auto simplices = gCoordTree.knn(point, kNeighbours);
	
	std::vector<float> g(3);
	int simplex = -1;
	for (auto &s : simplices){
		std::fill(g.begin(), g.end(), 0.);
		for (unsigned int i = 0; i < 9; i++){
			g[i / 3] += gInverseMatrices[s][i] * point[i % 3];
		}
	    if (std::all_of(g.cbegin(), g.cend(), [](float x) { return x >= 0; })){
	    	simplex = s;
	    	break;
	    }
	}
	
	if (simplex < 0){
		gErrorCounts++;
		rt_printf("The %d-th times cannot find positive weights!\n", gErrorCounts);
		return;
	}
	
	float sum = g[0] + g[1] + g[2];
	std::for_each(g.begin(), g.end(), [sum](float &x){	x /= sum;});
	
	std::vector<float> interpolation(gHRIRSamples, 0.0);
	for (unsigned int i = 0; i < 3; i++){
		std::transform(interpolation.cbegin(), interpolation.cend(), gHRIRInterleaved[gSimplices[simplex][i] * 2].cbegin(), 
					   interpolation.begin(), [g, i](const float &out, const float &in){	return out + in * g[i];});
	}
	std::copy(interpolation.begin(), interpolation.end(), gSampledHRIR[0].begin());
	
	std::fill(interpolation.begin(), interpolation.end(), 0.0);
	for (unsigned int i = 0; i < 3; i++){
		std::transform(interpolation.cbegin(), interpolation.cend(), gHRIRInterleaved[gSimplices[simplex][i] * 2 + 1].cbegin(), 
					   interpolation.begin(), [g, i](const float &out, const float &in){	return out + in * g[i];});
	}
	std::copy(interpolation.begin(), interpolation.end(), gSampledHRIR[1].begin());
}


// This function runs in an auxiliary task on Bela, calling process_fft
void nearest_hrir_background(void *)
{
	std::vector<float> point {gCachedX, gCachedY, gCachedZ};
	auto index = gCoordTree.nearest_index(point);
	
	rt_printf("%d \n", index);
	
	std::copy(gHRIRInterleaved[index * 2].begin(), gHRIRInterleaved[index * 2].end(), gSampledHRIR[0].begin());
	std::copy(gHRIRInterleaved[index * 2 + 1].begin(), gHRIRInterleaved[index * 2 + 1].end(), gSampledHRIR[1].begin());
}


bool setup(BelaContext *context, void *userData)
{	
	if (context->audioOutChannels != 2){
		rt_printf("Number of channels should be 2!");
    	return false;		
	}
	// Load the audio file
	gSoundBuffer = AudioFileUtilities::loadMono(gSoundFilename);
	
	if(gSoundBuffer.empty()) {
    	rt_printf("Error loading audio file '%s'\n", gSoundFilename.c_str());
    	return false;		
	}
	
	rt_printf("Loaded the audio file '%s' with %d frames (%.1f seconds)\n",
                        gSoundFilename.c_str(), gSoundBuffer.size(),
                        gSoundBuffer.size() / context->audioSampleRate);
	
	gOutputBuffer[0].resize(kBufferSize);
	gOutputBuffer[1].resize(kBufferSize);
	
	auto infoFile = gHRIRDir + "/" + kInfoFile;
	float x, y, z;
	int n;
	std::string line;
	std::fstream fs;
	std::stringstream parse;
	
	fs.open(infoFile, std::fstream::in);
    if (fs.fail()) {
        rt_printf("Cannot open '%s'\n", infoFile.c_str());
        return false;
    }
    
    while(std::getline(fs, line)){
    	parse << line;
    	parse >> x >> y >> z;
    	if (gRadius == 0){
    		gRadius = sqrt(x * x + y * y + z * z);
    	}
    	gHRIRCoordinates.push_back({x / gRadius, y / gRadius, z / gRadius});
    	std::vector<int> neighbours;
    	while(parse >> n)
    		neighbours.push_back(n);
    	gNeighborSimplices.push_back(neighbours);
    	parse.clear();
    }
    fs.close();
    
    auto simplexFile = gHRIRDir + "/" + kSimplexFile;
	fs.open(simplexFile, std::fstream::in);
    if (fs.fail()) {
        rt_printf("Cannot open '%s'\n", simplexFile.c_str());
        return false;
    }
	
	float w;   
	int s1, s2, s3;
    while(std::getline(fs, line)){
    	parse << line;
    	parse >> s1 >> s2 >> s3;
    	gSimplices.push_back({s1, s2, s3});
    	
    	std::vector<float> weights;
    	while(parse >> w)
    		weights.push_back(w);
    	gInverseMatrices.push_back(weights);
    	parse.clear();
    }
    
    
    
    
    for (auto &simplex : gSimplices){
    	std::vector<float> centre(3, 0.0);
    	for(auto i : simplex){
		    std::transform(centre.cbegin(), centre.cend(), gHRIRCoordinates[i].cbegin(),
    					   centre.begin(), std::plus<float>{});
    	}
    	auto tmp = centre;
		std::for_each(tmp.begin(), tmp.end(), [](float &x){	x *= x;});
		float norm = std::accumulate(tmp.begin(), tmp.end(), 0);
		float r = sqrt(norm);
		std::for_each(centre.begin(), centre.end(), [r](float &x){	x /= r;});
		gSimplicesCentres.push_back(centre);
    }
    
    
    
    rt_printf("%d %d %d %d\n", gNeighborSimplices.size(), gSimplices.size(), gInverseMatrices.size(), gHRIRCoordinates.size());
   
    
    for(int n = 0; n < gHRIRCoordinates.size(); n++){
    	auto hrirFile = gHRIRDir + "/" + std::to_string(n) + ".wav";
    	auto hrir = AudioFileUtilities::load(hrirFile);
    	// hrir[0].resize(64);
    	// hrir[1].resize(64);
    	if (n == 0){
    		gHRIRSamples = hrir[0].size();
    		gSampledHRIR[0].resize(gHRIRSamples);
    		gSampledHRIR[1].resize(gHRIRSamples);
    		gCurrentHRIR[0].resize(gHRIRSamples);
    		gCurrentHRIR[1].resize(gHRIRSamples);
    		gHRIRInterpolationGrad[0].resize(gHRIRSamples);
    		gHRIRInterpolationGrad[1].resize(gHRIRSamples);
    	}
    	gHRIRInterleaved.push_back(hrir[0]);
    	gHRIRInterleaved.push_back(hrir[1]);
    }
    
    
    gCoordTree = KDTree(gSimplicesCentres);
    // gCoordTree = KDTree(gHRIRCoordinates);
    
    
	// Set up the GUI
	gGui.setup(context->projectName);
	gGuiController.setup(&gGui, "HRTF Audio Player");	
	
	// Arguments: name, default value, minimum, maximum, increment
	gGuiController.addSlider("Elevation", 0, -90, 90, 0);
	gGuiController.addSlider("Azimuth", 0, -180, 180, 0);
	gGuiController.addSlider("Radius", gRadius, 0.3, 2, 0);
	
	// Set up the oscilloscope
	gScope.setup(3, context->audioSampleRate);
	
	// Set up the thread for the HRIR
	gSampleRetrieveTask = Bela_createAuxiliaryTask(weighted_hrir_background, 90, "bela-retrieve-hrir");

	return true;
}

void render(BelaContext *context, void *userData)
{
	float el = gGuiController.getSliderValue(0);
	float az = gGuiController.getSliderValue(1);
	float radius = gGuiController.getSliderValue(2);
	
	float col = map(90 - el, 0, 180, 0, M_PI);
	az = -map(az, -180, 180, -M_PI, M_PI);
	
	float col_sin = sin(col);
	float col_cos = cos(col);
	float az_sin = sin(az);
	float az_cos = cos(az);
	gX = col_sin * az_cos;
	gY = col_sin * az_sin;
	gZ = col_cos;
	
	float scaler = gRadius / radius;
	
    for(unsigned int n = 0; n < context->audioFrames; n++) {
    	float in = gSoundBuffer[gReadPointer] * scaler;
    	
		// Increment pointer into audio file buffer
        if(++gReadPointer >= gSoundBuffer.size())
        	gReadPointer = 0;
       
		
		for(unsigned int c = 0; c < 2; c++)
			std::transform(gCurrentHRIR[c].cbegin(), gCurrentHRIR[c].cend(), gHRIRInterpolationGrad[c].cbegin(), gCurrentHRIR[c].begin(), std::plus<float>());
		
		if (++gHopCounter >= kHopSize){
			gHopCounter = 0;
			for(unsigned int c = 0; c < 2; c++)
				std::transform(gCurrentHRIR[c].cbegin(), gCurrentHRIR[c].cend(), gSampledHRIR[c].cbegin(), 
							   gHRIRInterpolationGrad[c].begin(), [](const float &cur, const float &next){	return (next - cur) / kHopSize;});
			
			if ((gCachedX != gX) || (gCachedY != gCachedY) || (gCachedZ != gZ)){
				gCachedX = gX;
				gCachedY = gY;
				gCachedZ = gZ;
				Bela_scheduleAuxiliaryTask(gSampleRetrieveTask);	
			}
		}
		
		for (unsigned int c = 0; c < context->audioOutChannels; c++){
			float out = 0;
			gOutputBuffer[c][gOutputBufferPointer] = in;
			for (unsigned int k = 0; k < gHRIRSamples; k++)
				// gOutputBuffer[c][(gOutputBufferPointer + k) % kBufferSize] += gCurrentHRIR[c][k] * in;
				out += gCurrentHRIR[c][k] * gOutputBuffer[c][(gOutputBufferPointer - k + kBufferSize) % kBufferSize];
	        // audioWrite(context, n, c, gOutputBuffer[c][gOutputBufferPointer]);
	        // gOutputBuffer[c][gOutputBufferPointer] = 0;
	        audioWrite(context, n, c, out);
		}
		
        gOutputBufferPointer++;
		if (gOutputBufferPointer >= kBufferSize)
			gOutputBufferPointer = 0;
    	
    	// Write the output to the oscilloscope
    	gScope.log(in, in, in);
    }
}

void cleanup(BelaContext *context, void *userData)
{
	
}
