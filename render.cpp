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
#include <libraries/Gui/Gui.h>
#include <libraries/GuiController/GuiController.h>
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

// Buffers to hold HRIRs
std::vector<std::vector<float>> gHRIRInterleaved;
std::vector<std::vector<float>> gHRIRCoordinates;
std::vector<std::vector<float>> gSimplicesCentres;
std::vector<std::vector<int>> gNeighborSimplices;
std::vector<std::vector<int>> gSimplices;
std::vector<std::vector<float>> gInverseMatrices;

// Buffers to hold the sampled HRIRs
std::vector<float> gSampledHRIR[2];
std::vector<float> gCurrentHRIR[2];
std::vector<float> gHRIRInterpolationGrad[2];

// Error counts
int gErrorCounts = 0;

// KD tree for sample retrieval
KDTree gCoordTree;

// Coordinates of the sound source
float gX, gY, gZ;

// HRIR specifications
int gHRIRTruncatedSamples;
float gRadius = 0;

// Browser-based GUI to adjust parameters
Gui gGui;
GuiController gGuiController;

// Thread for HRIR processing
AuxiliaryTask gSampleRetrieveTask;

// Cached variables for the second thread
float gCachedX, gCachedY, gCachedZ;

void weighted_hrir_background(void *)
{
	std::vector<float> point{gCachedX, gCachedY, gCachedZ};
	auto simplices = gCoordTree.knn(point, kNeighbours);

	std::vector<float> g(3);
	int simplex = -1;
	for (auto &s : simplices)
	{
		std::fill(g.begin(), g.end(), 0.);
		for (unsigned int i = 0; i < 9; i++)
		{
			g[i / 3] += gInverseMatrices[s][i] * point[i % 3];
		}
		if (std::all_of(g.cbegin(), g.cend(), [](float x)
						{ return x >= 0; }))
		{
			simplex = s;
			break;
		}
	}

	if (simplex < 0)
	{
		gErrorCounts++;
		rt_printf("The %d-th times cannot find positive weights!\n", gErrorCounts);
		return;
	}

	auto interpolationIndexes = gSimplices[simplex];

	float sum = g[0] + g[1] + g[2];
	std::for_each(g.begin(), g.end(), [sum](float &x)
				  { x /= sum; });

	std::vector<float> weightedIR(gHRIRTruncatedSamples);

	for (unsigned int c = 0; c < 2; c++)
	{
		std::fill(weightedIR.begin(), weightedIR.end(), 0.0);
		for (unsigned int i = 0; i < 3; i++)
		{
			std::transform(weightedIR.cbegin(), weightedIR.cend(), gHRIRInterleaved[interpolationIndexes[i] * 2 + c].cbegin(),
						   weightedIR.begin(), [g, i](const float &out, const float &in)
						   { return out + in * g[i]; });
		}
		std::copy(weightedIR.begin(), weightedIR.end(), gSampledHRIR[c].begin());
	}
}

void nearest_hrir_background(void *)
{
	std::vector<float> point{gCachedX, gCachedY, gCachedZ};
	auto index = gCoordTree.nearest_index(point);

	std::copy(gHRIRInterleaved[index * 2].begin(), gHRIRInterleaved[index * 2].end(), gSampledHRIR[0].begin());
	std::copy(gHRIRInterleaved[index * 2 + 1].begin(), gHRIRInterleaved[index * 2 + 1].end(), gSampledHRIR[1].begin());
}

bool setup(BelaContext *context, void *userData)
{
	if ((context->audioOutChannels != 2) || (context->audioInChannels != 2))
	{
		rt_printf("Number of channels should be 2!");
		return false;
	}

	gOutputBuffer[0].resize(kBufferSize);
	gOutputBuffer[1].resize(kBufferSize);

	auto infoFile = gHRIRDir + "/" + kInfoFile;
	float x, y, z;
	int n;
	std::string line;
	std::fstream fs;
	std::stringstream parse;

	fs.open(infoFile, std::fstream::in);
	if (fs.fail())
	{
		rt_printf("Cannot open '%s'\n", infoFile.c_str());
		return false;
	}

	while (std::getline(fs, line))
	{
		parse << line;
		parse >> x >> y >> z;
		float r = sqrt(x * x + y * y + z * z);
		if (gRadius == 0)
		{
			gRadius = r;
		}
		gHRIRCoordinates.push_back({x / r, y / r, z / r});
		std::vector<int> neighbours;
		while (parse >> n)
			neighbours.push_back(n);
		gNeighborSimplices.push_back(neighbours);
		parse.clear();
	}
	fs.close();

	auto simplexFile = gHRIRDir + "/" + kSimplexFile;
	fs.open(simplexFile, std::fstream::in);
	if (fs.fail())
	{
		rt_printf("Cannot open '%s'\n", simplexFile.c_str());
		return false;
	}

	float w;
	int s1, s2, s3;
	while (std::getline(fs, line))
	{
		parse << line;
		parse >> s1 >> s2 >> s3;
		gSimplices.push_back({s1, s2, s3});

		std::vector<float> weights;
		while (parse >> w)
			weights.push_back(w);
		gInverseMatrices.push_back(weights);
		parse.clear();
	}
	fs.close();

	for (auto &simplex : gSimplices)
	{
		std::vector<float> centre(3, 0.0);
		for (auto i : simplex)
		{
			std::transform(centre.cbegin(), centre.cend(), gHRIRCoordinates[i].cbegin(),
						   centre.begin(), std::plus<float>{});
		}
		auto tmp = centre;
		std::for_each(tmp.begin(), tmp.end(), [](float &x)
					  { x *= x; });
		float r = sqrt(std::accumulate(tmp.begin(), tmp.end(), 0));
		std::for_each(centre.begin(), centre.end(), [r](float &x)
					  { x /= r; });
		gSimplicesCentres.push_back(centre);
	}

	rt_printf("Total number of HRIRs: %d\n", gHRIRCoordinates.size());
	rt_printf("Total number of simplices: %d\n", gSimplices.size());

	for (int n = 0; n < gHRIRCoordinates.size(); n++)
	{
		auto hrirFile = gHRIRDir + "/" + std::to_string(n) + ".wav";
		auto hrir = AudioFileUtilities::load(hrirFile);
		gHRIRInterleaved.push_back(hrir[0]);
		gHRIRInterleaved.push_back(hrir[1]);
		if (n == 0)
			gHRIRTruncatedSamples = hrir[0].size();
	}

	for (unsigned int c = 0; c < 2; c++)
	{
		gSampledHRIR[c].resize(gHRIRTruncatedSamples);
		gCurrentHRIR[c].resize(gHRIRTruncatedSamples);
		gHRIRInterpolationGrad[c].resize(gHRIRTruncatedSamples);
	}

	// Set up the KD tree for sample retrieval
	gCoordTree = KDTree(gSimplicesCentres);
	// gCoordTree = KDTree(gHRIRCoordinates);

	// Set up the GUI
	gGui.setup(context->projectName);
	gGuiController.setup(&gGui, "HRTF Audio Player");

	// Arguments: name, default value, minimum, maximum, increment
	gGuiController.addSlider("Elevation", 0, -60, 90, 0);
	gGuiController.addSlider("Azimuth", 0, -180, 180, 0);
	gGuiController.addSlider("Radius", gRadius, 0.3, 2, 0);

	// Set up the thread for the HRIR interpolation
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

	for (unsigned int n = 0; n < context->audioFrames; n++)
	{
		// float in = gSoundBuffer[gReadPointer] * scaler;
		float in = (audioRead(context, n, 0) + audioRead(context, n, 1)) * scaler;
		gOutputBuffer[0][gOutputBufferPointer] = in;
		gOutputBuffer[1][gOutputBufferPointer] = in;

		for (unsigned int c = 0; c < 2; c++)
		{
			float out = 0.0;
			for (unsigned int k = 0; k < gHRIRTruncatedSamples; k++)
			{
				out += gCurrentHRIR[c][k] * gOutputBuffer[c][(gOutputBufferPointer - k + kBufferSize) % kBufferSize];
			}
			audioWrite(context, n, c, out);
		}

		if (++gHopCounter >= kHopSize)
		{
			gHopCounter = 0;
			for (unsigned int c = 0; c < 2; c++)
				std::transform(gCurrentHRIR[c].cbegin(), gCurrentHRIR[c].cend(), gSampledHRIR[c].cbegin(),
							   gHRIRInterpolationGrad[c].begin(), [](const float &cur, const float &next)
							   { return (next - cur) / kHopSize; });

			if ((gCachedX != gX) || (gCachedY != gCachedY) || (gCachedZ != gZ))
			{
				gCachedX = gX;
				gCachedY = gY;
				gCachedZ = gZ;
				Bela_scheduleAuxiliaryTask(gSampleRetrieveTask);
			}
		}

		for (unsigned int c = 0; c < 2; c++)
			std::transform(gCurrentHRIR[c].cbegin(), gCurrentHRIR[c].cend(), gHRIRInterpolationGrad[c].cbegin(), gCurrentHRIR[c].begin(), std::plus<float>());

		gOutputBufferPointer++;
		if (gOutputBufferPointer >= kBufferSize)
			gOutputBufferPointer = 0;
	}
}

void cleanup(BelaContext *context, void *userData)
{
}
