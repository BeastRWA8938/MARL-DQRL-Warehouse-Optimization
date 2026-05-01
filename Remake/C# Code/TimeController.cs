using UnityEngine;

public class TimeController : MonoBehaviour
{
    [Header("Time Settings")]
    [Range(0.1f, 20f)]
    public float timeScale = 10.0f;

    void Update()
    {
        // Application.isBatchMode is true when running headless (no_graphics=True)
        if (Application.isBatchMode) 
        {
            return; // Do nothing, let ML-Agents control the speed!
        }

        Time.timeScale = timeScale;
    }

    // OnGUI creates a quick, functional UI on the screen without needing a Canvas
    private void OnGUI()
    {
        // Define the box dimensions and position (Top Left)
        int width = 300;
        int height = 110;
        int x = 20;
        int y = 20;

        // Draw the background box
        GUI.Box(new Rect(x, y, width, height), "Time Scale Controller");

        // Draw the text showing the current speed
        GUI.Label(new Rect(x + 20, y + 30, 260, 20), $"Current Speed: {timeScale:F2}x");

        // Draw the interactive slider (min: 0.1x, max: 20x)
        timeScale = GUI.HorizontalSlider(new Rect(x + 20, y + 55, 260, 20), timeScale, 0.1f, 20f);

        // Add some quick-snap buttons below the slider
        if (GUI.Button(new Rect(x + 20, y + 75, 80, 20), "1x (Normal)")) timeScale = 1.0f;
        if (GUI.Button(new Rect(x + 110, y + 75, 80, 20), "10x (Fast)")) timeScale = 10.0f;
        if (GUI.Button(new Rect(x + 200, y + 75, 80, 20), "20x (Max)")) timeScale = 20.0f;
    }
}