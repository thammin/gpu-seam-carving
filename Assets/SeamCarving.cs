using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.UI;

public class SeamCarving : MonoBehaviour
{
    public ComputeShader shader;
    public Texture source;
    public RawImage image;
    public bool showEnergyMap = false;
    public bool showSeam = true;
    public int count = 100;

    private RenderTexture _renderTexture;
    private ComputeBuffer _energyMap;
    private ComputeBuffer _sumOffset;
    private ComputeBuffer _optimalSeam;

    void Awake()
    {
        var width = source.width;
        var height = source.height;

        _renderTexture = new RenderTexture(width, height, 0, GraphicsFormat.R8G8B8A8_UNorm);
        _renderTexture.enableRandomWrite = true;

        _energyMap = new ComputeBuffer(width * height, sizeof(uint) * 2);
        _sumOffset = new ComputeBuffer(width * height, sizeof(uint) * 2);
        _optimalSeam = new ComputeBuffer(height, sizeof(uint));
    }

    async Task Start()
    {
        Graphics.Blit(source, _renderTexture);
        image.texture = _renderTexture;
        shader.SetBool("ShowEnergyMap", showEnergyMap);

        await Task.Delay(100);
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        await Execute();

        stopwatch.Stop();
        if (!showSeam) Debug.Log($"{stopwatch.Elapsed.TotalMilliseconds} ms");
    }

    public async Task Execute()
    {
        var N = showEnergyMap ? 1 : count;

        for (var i = 0; i < N; i++)
        {
            await Resize(i);
        }
    }

    async Task Resize(int step)
    {
        var imageSize = new Vector4(source.width - step, source.height, 1, 1);
        shader.SetInt("SizeX", (int)imageSize.x);
        shader.SetInt("SizeY", (int)imageSize.y);

        CalculateEnegry(imageSize);

        if (showEnergyMap) return;

        CalculateIndexMap(imageSize);
        CalculateEnergySum(imageSize);
        FindOptimalSeam(imageSize);

        if (showSeam)
        {
            DrawSeam(imageSize);
            await Task.Delay(16);
        }

        ResizeResult(imageSize);
    }

    void CalculateEnegry(Vector4 imageSize)
    {
        var kernel = shader.FindKernel("CalculateEnergy");
        shader.SetTexture(kernel, "Result", _renderTexture);
        shader.SetBuffer(kernel, "EnergyMap", _energyMap);
        shader.SetBuffer(kernel, "SumOffset", _sumOffset);

        var dispatch = GetDispatchCount(kernel, imageSize);
        shader.Dispatch(kernel, dispatch.x, dispatch.y, dispatch.z);
    }

    void CalculateIndexMap(Vector4 imageSize)
    {
        var kernel = shader.FindKernel("CalculateIndexMap");
        shader.SetBuffer(kernel, "EnergyMap", _energyMap);
        shader.SetBuffer(kernel, "SumOffset", _sumOffset);

        var dispatch = GetDispatchCount(kernel, imageSize);
        shader.Dispatch(kernel, dispatch.x, dispatch.y, dispatch.z);
    }

    void CalculateEnergySum(Vector4 imageSize)
    {
        var kernel = shader.FindKernel("CalculateEnergySum");
        shader.SetTexture(kernel, "Result", _renderTexture);
        shader.SetBuffer(kernel, "EnergyMap", _energyMap);
        shader.SetBuffer(kernel, "SumOffset", _sumOffset);

        var dispatch = GetDispatchCount(kernel, imageSize);
        for (var i = 0; i < Mathf.Log(imageSize.y, 2); i++)
        {
            shader.SetInt("EnergySumStep", i + 1);
            shader.Dispatch(kernel, 1, dispatch.y, 1);
        }
    }

    void FindOptimalSeam(Vector4 imageSize)
    {
        var kernel = shader.FindKernel("FindOptimalSeam");
        var dispatch = GetDispatchCount(kernel, imageSize);
        shader.SetBuffer(kernel, "EnergyMap", _energyMap);
        shader.SetBuffer(kernel, "SumOffset", _sumOffset);
        shader.SetBuffer(kernel, "OptimalSeam", _optimalSeam);
        shader.Dispatch(kernel, 1, dispatch.y, 1);
    }

    void DrawSeam(Vector4 imageSize)
    {
        var kernel = shader.FindKernel("DrawSeam");
        var dispatch = GetDispatchCount(kernel, imageSize);
        shader.SetTexture(kernel, "Result", _renderTexture);
        shader.SetBuffer(kernel, "OptimalSeam", _optimalSeam);
        shader.Dispatch(kernel, 1, dispatch.y, 1);
    }

    void ResizeResult(Vector4 imageSize)
    {
        var kernel = shader.FindKernel("ResizeResult");
        var dispatch = GetDispatchCount(kernel, imageSize);
        shader.SetTexture(kernel, "Result", _renderTexture);
        shader.SetBuffer(kernel, "OptimalSeam", _optimalSeam);
        shader.Dispatch(kernel, 1, dispatch.y, 1);
    }

    Vector3Int GetDispatchCount(int kernel, Vector4 imageSize)
    {
        shader.GetKernelThreadGroupSizes(kernel, out uint x, out uint y, out uint z);

        return new Vector3Int(
            Mathf.CeilToInt(imageSize.x / x),
            Mathf.CeilToInt(imageSize.y / y),
            Mathf.CeilToInt(imageSize.z / z)
        );
    }

    void OnDestroy()
    {
        _renderTexture.Release();
        _energyMap.Release();
        _sumOffset.Release();
        _optimalSeam.Release();
    }
}
