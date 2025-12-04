#ifndef RASTERIZER_H
#define RASTERIZER_H

#include "CoreLib/Basic.h"
#include "ProjectedTriangle.h"
#include "RenderState.h"
#include <smmintrin.h>
namespace RasterRenderer
{

    struct TriangleSIMD
    {
        // 1 if triangle "owns" edge, 0 otherwise
        // Samples lying on "owned" edges are in the triangle.
        // Samples lying on non-owned edges are not in the triangle
        int isOwnerEdge[3];

        __m128 invArea;
        __m128 z0, dzdx, dzdy;  // depth plane equation
        __m128i a0, a1, a2, b0, b1, b2; // edge equations
        __m128i x0, y0, x1, y1, x2, y2; // vertex positions (N.4 format)

        inline void LoadForCoordinates(ProjectedTriangle & tri)
        {
            a0 = _mm_set1_epi32(tri.A0);
            a1 = _mm_set1_epi32(tri.A1);
            a2 = _mm_set1_epi32(tri.A2);
            b0 = _mm_set1_epi32(tri.B0);
            b1 = _mm_set1_epi32(tri.B1);
            b2 = _mm_set1_epi32(tri.B2);
            x0 = _mm_set1_epi32(tri.X0);
            y0 = _mm_set1_epi32(tri.Y0);
            x1 = _mm_set1_epi32(tri.X1);
            y1 = _mm_set1_epi32(tri.Y1);
            x2 = _mm_set1_epi32(tri.X2);
            y2 = _mm_set1_epi32(tri.Y2);
            z0 = _mm_set1_ps(tri.fZ0);
            dzdx = _mm_set1_ps(tri.fDZDX);
            dzdy = _mm_set1_ps(tri.fDZDY);
            invArea = _mm_set1_ps(tri.InvArea);
        }

        inline void Load(ProjectedTriangle & tri)
        {
            a0 = _mm_set1_epi32(tri.A0);
            a1 = _mm_set1_epi32(tri.A1);
            a2 = _mm_set1_epi32(tri.A2);
            b0 = _mm_set1_epi32(tri.B0);
            b1 = _mm_set1_epi32(tri.B1);
            b2 = _mm_set1_epi32(tri.B2);
            x0 = _mm_set1_epi32(tri.X0);
            y0 = _mm_set1_epi32(tri.Y0);
            x1 = _mm_set1_epi32(tri.X1);
            y1 = _mm_set1_epi32(tri.Y1);
            x2 = _mm_set1_epi32(tri.X2);
            y2 = _mm_set1_epi32(tri.Y2);
            z0 = _mm_set1_ps(tri.fZ0);
            dzdx = _mm_set1_ps(tri.fDZDX);
            dzdy = _mm_set1_ps(tri.fDZDY);
            invArea = _mm_set1_ps(tri.InvArea);

            isOwnerEdge[0] = tri.Y0 < tri.Y1 || (tri.Y0 == tri.Y1 && tri.Y2 >= tri.Y0);
            isOwnerEdge[1] = tri.Y1 < tri.Y2 || (tri.Y1 == tri.Y2 && tri.Y0 >= tri.Y1);
            isOwnerEdge[2] = tri.Y2 < tri.Y0 || (tri.Y0 == tri.Y2 && tri.Y1 >= tri.Y0);
        }

        // evaluate Z at given sample point
        inline __m128 GetZ(__m128i x, __m128i y)
        {
            auto dx = _mm_sub_epi32(x, x0);
            auto dy = _mm_sub_epi32(y, y0);
            return _mm_add_ps(z0, _mm_add_ps(_mm_mul_ps(dzdx, _mm_cvtepi32_ps(dx)), _mm_mul_ps(dzdy, _mm_cvtepi32_ps(dy))));
        }

        // evaluate barycentric coordinates at given sample point
        inline void GetCoordinates(__m128 &w0, __m128 &w1, __m128 &w2, __m128i x, __m128i y)
        {
            __m128i iw0 = _mm_add_epi32(_mm_mullo_epi32(a0, _mm_sub_epi32(x, x0)), _mm_mullo_epi32(b0, _mm_sub_epi32(y, y0)));
            __m128i iw1 = _mm_add_epi32(_mm_mullo_epi32(a1, _mm_sub_epi32(x, x1)), _mm_mullo_epi32(b1, _mm_sub_epi32(y, y1)));
            __m128i iw2 = _mm_add_epi32(_mm_mullo_epi32(a2, _mm_sub_epi32(x, x2)), _mm_mullo_epi32(b2, _mm_sub_epi32(y, y2)));
            w0 = _mm_mul_ps(_mm_cvtepi32_ps(iw0), invArea);
            w1 = _mm_mul_ps(_mm_cvtepi32_ps(iw1), invArea);
            w2 = _mm_mul_ps(_mm_cvtepi32_ps(iw2), invArea);
        }

        // TestQuadFragment function: returns a bit mask indicating whether the sample for each fragment is covered
        // (assumes one sample per fragment == no MSAA support):
        //
        // Parameters:
        //
        //    x: the x coordinates for the four samples in the quad fragment (in N.4 fixed-point format)
        //    y: the y coordinates for the four samples in the quad fragment (in N.4 fixed-point format)
        //
        // Tips:
        //
        //   This function tests four sample points at a time, defined by (x[i], y[i]).
        //   the return value is a 32 bit integer, where bit 3 (0x8), 7(0x80), 11(0x800), 15(0x8000)
        //   represents whether sample 0, 1, 2, 3 is covered.
        //
        //   You can implement the function directly using SSE intrinsics, or you can extract
        //   the x,y argument into int[4] and do the computation four times, one for each sample,
        //   and then repack the results into a bitmask with the format described above.
        //
        //   to extract __m128 x into elements of the array: float x_extract[4], use the following code:
        //
        //      CORE_LIB_ALIGN_16(float x_extract[4]);
        //      _mm_store_ps(x_extract, x);
        //
        //   to extract __m128i xi into elements of the array: int x_extract[4], use the following code:
        //
        //      CORE_LIB_ALIGN_16(int x_extract[4]);
        //      _mm_store_ps((float*)x_extract, _mm_castsi128_ps(xi));

        inline int TestQuadFragment(__m128i x, __m128i y)
        {
            // Implement triangle-quad-fragment coverage testing here.
            // Return an integer encoding the coverage testing results.

            /******************************
             * Ei = A(x-xi) + B(y-yi)
             ******************************/
            __m128i x_diff_0 = _mm_sub_epi32(x, x0);
            __m128i x_diff_1 = _mm_sub_epi32(x, x1);
            __m128i x_diff_2 = _mm_sub_epi32(x, x2);

            __m128i y_diff_0 = _mm_sub_epi32(y, y0);
            __m128i y_diff_1 = _mm_sub_epi32(y, y1);
            __m128i y_diff_2 = _mm_sub_epi32(y, y2);

            __m128i ax_product_0 = _mm_mullo_epi32(a0, x_diff_0);
            __m128i by_product_0 = _mm_mullo_epi32(b0, y_diff_0);
            __m128i e0 = _mm_add_epi32(ax_product_0, by_product_0);

            __m128i ax_product_1 = _mm_mullo_epi32(a1, x_diff_1);
            __m128i by_product_1 = _mm_mullo_epi32(b1, y_diff_1);
            __m128i e1 = _mm_add_epi32(ax_product_1, by_product_1);

            __m128i ax_product_2 = _mm_mullo_epi32(a2, x_diff_2);
            __m128i by_product_2 = _mm_mullo_epi32(b2, y_diff_2);
            __m128i e2 = _mm_add_epi32(ax_product_2, by_product_2);

            /*********************************************
             * Extract, then compare E0, E1, and E2 to 0,
             * pixel by pixel
             *********************************************/
            int final_mask = 0x0000;

            CORE_LIB_ALIGN_16(int e0_extract[4]);
            _mm_store_ps((float*)e0_extract, _mm_castsi128_ps(e0));

            CORE_LIB_ALIGN_16(int e1_extract[4]);
            _mm_store_ps((float*)e1_extract, _mm_castsi128_ps(e1));

            CORE_LIB_ALIGN_16(int e2_extract[4]);
            _mm_store_ps((float*)e2_extract, _mm_castsi128_ps(e2));
            // Test pixel by pixel for success rather than e by e
            for (int i = 0; i < 4; i++) {
                bool e0pass, e1pass, e2pass;
                e0pass = e0_extract[i] > 0 || (e0_extract[i] == 0 && isOwnerEdge[0]);
                e1pass = e1_extract[i] > 0 || (e1_extract[i] == 0 && isOwnerEdge[1]);
                e2pass = e2_extract[i] > 0 || (e2_extract[i] == 0 && isOwnerEdge[2]);

                if (e0pass && e1pass && e2pass) final_mask |= 0x8 << (4 * i);
            }

            return final_mask;

        }
    };

    // RasterizeTriangle function: conservatively generate quad fragments that are potentially covered by a triangle.
    // the function takes pixel bounds as input (regionX0, regionY0, regionW, regionH) and should not generate quad
    // fragments outside the given bounds.
    //
    // Parameters:
    //
    //   tileX0, tileY0: the top-left corner coordinate (in pixels) of current working tile
    //   (let's assume we're using regionX0 and regionY0)
    //   tileW, tileH: the width and height of current working tile, in pixels
    //   (same deal here, regionW and regionH)
    //   tri: setup triangle equations (see ProjectedTriangle.h)
    //   triSIMD: all values of tri in SIMD registers (see struct definition above)
    //   processQuadFragmentFunc: for every quad fragment that may generate coverage, this method should
    //   call processQuadFragmentFunc
    //
    // Tips:
    //
    //   (regionX0, regionX0) is the bottom left of the pixel bounds
    //   for every 2x2 pixel region that may be covered by the
    //   triangle (given by bottom-left pixel corner (x,y), this
    //   method should call:
    //
    //      processQuadFragmentFunc(x, y, trivialAccept);
    //
    //   'trivialAccept' should be set to true if you'd like to
    //   notify subsequent quad fragment processing logic that the
    //   quad fragment is entirely covered by the triangle.  That
    //   is, that all samples in the quad fragment are covered.
    //
    //   If you don't need to perform this optimization in your renderer,
    //   always set trivialAccept to false.

    template<typename ProcessPixelFunc>
    inline void RasterizeTriangle(int regionX0, int regionY0, int regionW, int regionH, const ProjectedTriangle &tri, TriangleSIMD& triSIMD, ProcessPixelFunc processQuadFragmentFunc)
    {

        // Implement triangle rasterization here.

        // use min/max coordinates + bounding box from Projected Triangle
        int xMin = std::min(std::min(tri.X0, tri.X1), tri.X2);
        int xMax = std::max(std::max(tri.X0, tri.X1), tri.X2);
        int yMin = std::min(std::min(tri.Y0, tri.Y1), tri.Y2);
        int yMax = std::max(std::max(tri.Y0, tri.Y1), tri.Y2);

        xMin >>= 4;
        xMax >>= 4;
        yMin >>= 4;
        yMax >>= 4;

        xMin -= 1;
        xMax += 1;
        yMin -= 1;
        yMax += 1;

        if (xMin < regionX0) xMin = regionX0;
        if (xMax > regionX0 + regionW) xMax = regionX0 + regionW;
        if (yMin < regionY0) yMin = regionY0;
        if (yMax > regionY0 + regionH) yMax = regionY0 + regionH;

        for (int tileY = yMin; tileY < yMax; tileY += 2) {
            for (int tileX = xMin; tileX < xMax; tileX += 2) {
                processQuadFragmentFunc(tileX, tileY, false);
            }
        }

    }
}

#endif
