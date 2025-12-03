#include "RendererImplBase.h"
#include "CommonTraceCollection.h"
#include <algorithm>
#include <immintrin.h>
#include <vector>
#include <set>
#include <cassert>
#include <assert.h>

using namespace std;

namespace RasterRenderer
{
    class TiledRendererAlgorithm
    {
    private:
        static const int FragmentBufferSize = 65536;

        // starter code uses 32x32 tiles, feel free to change

        static const int Log2TileSize = 6;
        static const int TileSize = 1 << Log2TileSize;

        // HINT:
        // a compact representation of a frame buffer tile
        // Use if you wish (optional).
        // (What other data-structures might be necessary?)

        // tiles are how we partition the process stage, need a frameBuffer copy per tile
        std::vector<FrameBuffer> frameBufferTiles;

        // contain indices into projected triangle list
        std::vector<std::vector<std::vector<std::vector<int>>>> bins; // bins[row][col][coreId]

        // render target is grid of tiles: see SetFrameBuffer
        int gridWidth, gridHeight;
        FrameBuffer * frameBuffer;
    public:
        inline void Init()
        {

        }

        inline void Clear(const Vec4 & clearColor, bool color, bool depth)
        {
            for (auto & fb : frameBufferTiles)
                fb.Clear(clearColor, color, depth);
            frameBuffer->Clear(clearColor, color, depth);
            printf("Clear!\n");
        }

        inline void SetFrameBuffer(FrameBuffer * frameBuffer)
        {
            this->frameBuffer = frameBuffer;

            // compute number of necessary bins
            gridWidth = frameBuffer->GetWidth() >> Log2TileSize;
            gridHeight = frameBuffer->GetHeight() >> Log2TileSize;
            if (frameBuffer->GetWidth() & (TileSize - 1))
                gridWidth++;
            if (frameBuffer->GetHeight() & (TileSize - 1))
                gridHeight++;

            for(int y = 0; y < frameBuffer->GetHeight(); y++) {
                for(int x = 0; x < frameBuffer->GetWidth(); x++) {
                    frameBuffer->SetZ(x,y,0,FLT_MAX);
                }
            }

            // Implementations may want to allocation/initialize
            // any necessary data-structures here. You may wish to
            // consult the HINT at the top of the file.

            for (int g = 0; g < gridWidth * gridHeight; g++) {
                frameBufferTiles.emplace_back(frameBuffer->GetWidth(),
                                              frameBuffer->GetHeight(),
                                              frameBuffer->GetSampleCount());
                auto fb = frameBufferTiles[g];
                fb.SetSize(frameBuffer->GetWidth(), frameBuffer->GetHeight(),
                           frameBuffer->GetSampleCount());

                for (int y = 0; y < fb.GetHeight(); y++) {
                    for (int x = 0; x < fb.GetWidth(); x++) {
                        fb.SetZ(x, y, 0, FLT_MAX);
                    }
                }
            }

            for (int y = 0; y < gridHeight; y++) {
                bins.emplace_back();
                for (int x = 0; x < gridWidth; x++) {
                    bins[y].emplace_back();
                    for (int c = 0; c < Cores; c++) {
                        bins[y][x].emplace_back();
                    }
                }
            }
        }

        inline void Finish()
        {
            //printf("Finished?\n");
            // Finish() is called at the end of the frame. If it hasn't done so, your
            // implementation should flush local per-tile framebuffer contents to the
            // global frame buffer (this->frameBuffer) here.

            int fbWidth = frameBuffer->GetWidth();
            int fbHeight = frameBuffer->GetHeight();

            for (auto & fbtile : frameBufferTiles) {
                for (int y = 0; y < fbHeight; y++) {
                    for (int x = 0; x < fbWidth; x++) {
                        if (fbtile.GetZ(x, y, 0) < frameBuffer->GetZ(x, y, 0)) {
                            auto fbtilePix = fbtile.GetPixel(x, y, 0);
                            auto fbtileZ = fbtile.GetZ(x, y, 0);
                            frameBuffer->SetZ(x, y, 0, fbtileZ);
                            frameBuffer->SetPixel(x, y, 0, fbtilePix);
                        }
                    }
                }
            }
        }

        inline void BinTriangles(RenderState & state, ProjectedTriangleInput & input, int vertexOutputSize, int threadId)
        {

            auto & triangles = input.triangleBuffer[threadId];

            for (int i = 0; i < triangles.Count(); i++)
            {
                auto & tri = triangles[i];

                // Process triangles into bins here.
                // Keep in mind BinTriangles is called once per worker thread.
                // Each thread grabs the list of triangles is it responsible
                // for, and should bin them here.

                int triLeft = std::min(std::min(tri.X0, tri.X1), tri.X2) >> 4;
                triLeft = std::max(triLeft, 0);
                triLeft >>= Log2TileSize;
                int triRight = std::max(std::max(tri.X0, tri.X1), tri.X2) >> 4;
                triRight = std::min(triRight, frameBuffer->GetWidth() - 1);
                triRight >>= Log2TileSize;

                int triBottom = std::min(std::min(tri.Y0, tri.Y1), tri.Y2) >> 4;
                triBottom = std::min(triBottom, 0);
                triBottom >>= Log2TileSize;
                int triTop = std::max(std::max(tri.Y0, tri.Y1), tri.Y2) >> 4;
                triTop = std::min(triTop, frameBuffer->GetHeight() - 1);
                triTop >>= Log2TileSize;


                // DEBUG
                triBottom = 0;
                triTop = 0;
                triLeft = 0;
                triRight = 0;

                for (int y = triBottom; y <= triTop; y++) {
                    for (int x = triLeft; x <= triRight; x++) {
                        bins[y][x][threadId].emplace_back(i);
                    }
                }
            }
            printf("Core %i binned %i triangles\n", threadId, triangles.Count());
        }

        inline void ProcessBin(RenderState & state, ProjectedTriangleInput & input, int vertexOutputSize, int tileId)
        {
            // This thread should process the bin of triangles corresponding to
            // render target tile 'tileId'. You should take heavy inspiration from
            // 'RenderProjectedBatch' in 'NontiledForwardRenderer.cpp'

            // Keep in mind that in a tiled renderer, rasterization and fragment processing
            // is parallelized across tiles, not fragments within a tile.  Processing within a tile
            // is carried out sequentially. Thus your implementation will likely call ShadeFragment()
            // to shade a single quad fragment at a time, rather than call ShadeFragments()
            // as was done on the full fragment buffer in the non-tiled reference
            // implementation. (both functions are defined in RendererImplBase.h).
            //
            // Some example code is given below:
            //
            // let 'tri' be a triangle
            // let 'triangleId' be its position in the original input list
            // let 'triSIMD' be the triangle info loaded into SIMD registers (see reference impl)

            // get triangle barycentric coordinates:
            //     __m128 gamma, beta, alpha;
            //     triSIMD.GetCoordinates(gamma, alpha, beta, coordX_center, coordY_center);
            //     auto z = triSIMD.GetZ(coordX_center, coordY_center);
            //
            // call ShadeFragment to shade a single quad fragment, storing the output colors (4 float4's) in shadeResult:
            //     CORE_LIB_ALIGN_16(float shadeResult[16]);
            //     ShadeFragment(state, shadeResult, beta, gamma, alpha, triangleId, tri.ConstantId, VERTEX_BUFFER_FOR_TRIANGLE, vertexOutputSize, INDEX_BUFFER_FOR_TRIANGLE);

            // iterate through triangles, determine bin to do the above

            //debug
            /*if (tileId == 0) printf("Called!\n");
            return;*/

            FrameBuffer *myFrameBuffer = &(frameBufferTiles[tileId]);
            std::vector<std::vector<int>> myBins = bins[tileId / gridWidth][tileId % gridWidth];

            __m128 one = _mm_set_ps1(1.0f);

            static __m128i xOffset = _mm_set_epi32(24, 8, 24, 8);
            static __m128i yOffset = _mm_set_epi32(24, 24, 8, 8);

            int sampleCount = myFrameBuffer->GetSampleCount();
            int multiSampleLevel = myFrameBuffer->GetSampleCountLog2();

            //ProjectedTriangleInput::Iterator triIter(input);

            for (int i = 0; i < Cores; i++) {
                auto triangles = input.triangleBuffer[i];
                assert(myBins[i][myBins[i].size() - 1] < triangles.Count());
                if (tileId == 0) printf("Accessed triangles in core %i of %i\n", i, Cores - 1);
                // get the triangle
                for (int triId : myBins[i]) {
                    auto tri = triangles[triId];
                    TriangleSIMD triSIMD;
                    triSIMD.Load(tri);

                    // Start rasterization
                    int rasterLeft = TileSize * (tileId % gridWidth);
                    int rasterRight = min(rasterLeft + TileSize, frameBuffer->GetWidth());
                    int rasterTop = TileSize * (tileId / gridWidth);
                    int rasterBottom = min(rasterTop + TileSize, frameBuffer->GetHeight());
                    int rasterWidth = rasterRight - rasterLeft;
                    int rasterHeight = rasterTop - rasterBottom;

                    RasterizeTriangle(rasterLeft, rasterTop, rasterWidth, rasterHeight,
                                      tri, triSIMD, [&](int qfx, int qfy, bool trivialAccept)
                    {
                        // BruteForceRasterizer invokes this lambda once per
                        // "potentially covered" quad fragment.

                        // The quad fragment's base pixel coordinate
                        // is (qfx,qfy).  These are in units of pixels.

                        // trivialAccept is true if the rasterizer determined this
                        // quad fragment is a trivial accept case for the current
                        // triangle.

                        // coordX_center and coordY_center hold the
                        // pixel-center coordinates for the four pixels in this
                        // quad fragment. Coordinates are stored in an N.4
                        // fixed-point representation. (note shift-left by 4 of qfx and qfy,
                        // which convertes these values from pixel values to
                        // fixed-point values in N.4 format)
                        __m128i coordX_center, coordY_center;
                        coordX_center = _mm_add_epi32(_mm_set1_epi32(qfx << 4), xOffset);
                        coordY_center = _mm_add_epi32(_mm_set1_epi32(qfy << 4), yOffset);

                        int x = qfx;
                        int y = qfy;

                        // perform coverage test for all samples, if necessary
                        int coverageMask = trivialAccept ? 0xFFFF :
                                            triSIMD.TestQuadFragment(coordX_center, coordY_center);

                        //printf("%x\n", coverageMask);

                        // evaluate Z for each sample point
                        auto zValues = triSIMD.GetZ(coordX_center, coordY_center);

                        // copy z values out of SIMD register
                        CORE_LIB_ALIGN_16(float zStore[4]);
                        _mm_store_ps(zStore, zValues);

                        // THIS IS WHAT'S USED LATER
                        FragmentCoverageMask visibility;

                        // This is the early-Z check:
                        //
                        // As a result of early-Z the fragment's coverage mask is
                        // updated to have bits set only for covered samples that
                        // passed Z.  Later in the pipeline this mask is needed to
                        // know what frame buffer color samples to update.

                        if (coverageMask & 0x0008)
                        {
                            if (myFrameBuffer->GetZ(x, y, 0) > zStore[0])
                            {
                                visibility.SetBit(0);
                                myFrameBuffer->SetZ(x, y, 0, zStore[0]);
                            }
                        }
                        if (coverageMask & 0x0080)
                        {
                            if (myFrameBuffer->GetZ(x + 1, y, 0) > zStore[1])
                            {
                                visibility.SetBit(1);
                                myFrameBuffer->SetZ(x + 1, y, 0, zStore[1]);
                            }
                        }
                        if (coverageMask & 0x0800)
                        {
                            if (myFrameBuffer->GetZ(x, y + 1, 0) > zStore[2])
                            {
                                visibility.SetBit(2);
                                myFrameBuffer->SetZ(x, y + 1, 0, zStore[2]);
                            }
                        }
                        if (coverageMask & 0x8000)
                        {
                            if (myFrameBuffer->GetZ(x + 1, y + 1, 0) > zStore[3])
                            {
                                visibility.SetBit(3);
                                myFrameBuffer->SetZ(x + 1, y + 1, 0, zStore[3]);
                            }
                        }

                        // Shade this fragment if
                        // any sample is visible
                        if (visibility.Any())
                        {
                            // retrieve triangle barycentric coordinates at each
                            // sample point in stick them into the quad fragment.
                            // Renderer uses these coordinates to evaluate
                            // triangle attributes at the shading sample point
                            // during shading (e.g., to sample texture coordinates uv)
                            __m128 gamma, beta, alpha;
                            triSIMD.GetCoordinates(gamma, alpha, beta, coordX_center, coordY_center);
                            auto z = triSIMD.GetZ(coordX_center, coordY_center);

                            // ShadeFragment individually
                            CORE_LIB_ALIGN_16(float shadeResult[16]);

                            ShadeFragment(state, shadeResult, beta, gamma, alpha, triId, tri.ConstantId,
                                          input.vertexOutputBuffer[i].Buffer(), vertexOutputSize,
                                          input.indexOutputBuffer[i].Buffer());


                            if (visibility.GetBit(0))
                            {
                                myFrameBuffer->SetPixel(x, y, 0,
                                                        Vec4(shadeResult[0], shadeResult[4],
                                                             shadeResult[8], shadeResult[12]));
                            }
                            if (visibility.GetBit(1)) {
                                myFrameBuffer->SetPixel(x + 1, y, 0,
                                                        Vec4(shadeResult[1], shadeResult[5],
                                                             shadeResult[9], shadeResult[13]));
                            }
                            if (visibility.GetBit(2)) {
                                myFrameBuffer->SetPixel(x, y + 1, 0,
                                                        Vec4(shadeResult[2], shadeResult[6],
                                                             shadeResult[10], shadeResult[14]));
                            }
                            if (visibility.GetBit(3)) {
                                myFrameBuffer->SetPixel(x + 1, y + 1, 0,
                                                        Vec4(shadeResult[3], shadeResult[7],
                                                             shadeResult[11], shadeResult[15]));
                            }
                        }
                    });
                    //printf("Rasterized triangle %i in tile %i!\n", triId, tileId);
                    // next triangle
                }
                //if (tileId == 0) printf("Finished with %i triangles in Core %i\n", triangles.Count(), i);
            }

            //if (tileId == 0) printf("Done looping through\n");
        }



        inline void RenderProjectedBatch(RenderState & state, ProjectedTriangleInput & input, int vertexOutputSize)
        {
            bins = {};
            for (int y = 0; y < gridHeight; y++) {
                bins.emplace_back();
                for (int x = 0; x < gridWidth; x++) {
                    bins[y].emplace_back();
                    for (int c = 0; c < Cores; c++) {
                        bins[y][x].emplace_back();
                    }
                }
            }

            // Pass 1:
            //
            // The renderer is structured so that input a set of triangle lists
            // (exactly one list of triangles for each core to process).
            // As shown in BinTriangles() above, each thread bins the triangles in
            // input.triangleBuffer[threadId]
            //
            // Below we create one task per core (i.e., one thread per
            // core).  That task should bin all the triangles in the
            // list it is provided into bins: via a call to BinTriangles
            Parallel::For(0, Cores, 1, [&](int threadId)
            {
                BinTriangles(state, input, vertexOutputSize, threadId);
            });

            // Pass 2:
            //
            // process all the tiles created in pass 1. Create one task per
            // tile (not one per core), and distribute all the tasks among the cores.  The
            // third parameter to the Parallel::For call is the work
            // distribution granularity.  Increasing its value might reduce
            // scheduling overhead (consecutive tiles go to the same core),
            // but could increase load imbalance.  (You can probably leave it
            // at one.)
            Parallel::For(0, gridWidth*gridHeight, 1, [&](int tileId)
            {
                ProcessBin(state, input, vertexOutputSize, tileId);
                //printf("Processed tile %i\n", tileId);
            });
            //printf("All tiles processed.\n");
        }
    };

    IRasterRenderer * CreateTiledRenderer()
    {
        return new RendererImplBase<TiledRendererAlgorithm>();
    }
}
