using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class MusicLoopScript : MonoBehaviour
{
    public float BPM = 100;
    public bool isPlaying = false;

    public int beat;
    public int measure;

    private float timer;
    private float beatDuration;

    public AudioSource myAudioSource;

    public SpriteRenderer pauseButton;
    public Sprite pauseButtonRender;
    public Sprite playButtonRender;

    // File paths for each instrument track
    public AudioClip pianoClip1;
    public AudioClip pianoClip2;


    private AudioClip[] pianoClips;


    // The number of the sound clip to play. 0 stands for 'none'
    public int pianoTrack = 0;
    public int guitarTrack = 0;
    public int saxophoneTrack = 0;
    public int bassTrack = 0;


    // Start is called before the first frame update
    void Start()
    {
        beat = 0;
        measure = 1;
        timer = 0.0f;

        beatDuration = 60 / BPM;

        pianoClips = new AudioClip[] {null, pianoClip1, pianoClip2};
        //newBeat(1);
        DontDestroyOnLoad(gameObject);
        //SceneManager.LoadScene("Assets/Scenes/MainMenu.unity");
    }

    // Update is called once per frame
    void Update()
    {
        // Increment timer only if playing
        if (isPlaying) {
            timer += Time.deltaTime;
        }

        // Beat has finished, start a new beat
        if (timer >= beatDuration) {
            timer -= beatDuration;
            beat += 1;
            if (beat > 4) {
                measure += 1;
                beat = 1;
            }

            newBeat(beat);
            

        }
    }

    void newBeat(int beatNumber) {
        // Play a sound here
        //myAudioSource.PlayOneShot(pianoClips[0], 1);

        pianoBeat(beat);
        guitarBeat(beat);
        saxophoneBeat(beat);
        bassBeat(beat);


    }

    void pianoBeat(int beatNumber) {
        if (beatNumber == 1) {
            myAudioSource.PlayOneShot(pianoClips[pianoTrack], 1);
        }
    }

    void guitarBeat(int beatNumber) {
        
    }

    void saxophoneBeat(int beatNumber) {
        
    }

    void bassBeat(int beatNumber) {
        
    }


    [ContextMenu("Toggle Music")]
    public void toggleMusic() {
        if (isPlaying) {
            isPlaying = false;
            pauseButton.sprite = playButtonRender;
        }
        else {
            isPlaying = true;
            pauseButton.sprite = pauseButtonRender;
        }
    }


    public void changeInstrument() {
        SceneManager.LoadScene("Assets/Scenes/ChooseInstrument.unity");
    }


}
