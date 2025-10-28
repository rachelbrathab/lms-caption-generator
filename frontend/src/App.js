import React, { useState } from 'react';
import axios from 'axios';
import {
  Container, Box, Button, Typography, Paper, Grid,
  CircularProgress, IconButton, Tooltip, TextField
} from '@mui/material';
import { 
  UploadFile, 
  CopyAll, 
  VolumeUp, 
  Image as ImageIcon 
} from '@mui/icons-material';
import { createTheme, ThemeProvider, alpha } from '@mui/material/styles';

// Create a custom theme for our "glass" UI
const glassTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#f0f0f0',
    },
    background: {
      paper: 'rgba(255, 255, 255, 0.1)',
      default: 'transparent',
    },
    text: {
      primary: '#ffffff',
      secondary: '#e0e0e0',
    }
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
        },
        contained: {
          background: 'rgba(255, 255, 255, 0.2)',
          border: '1px solid rgba(255, 255, 255, 0.3)',
          '&:hover': {
            background: 'rgba(255, 255, 255, 0.3)',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            background: 'rgba(0, 0, 0, 0.2)',
            borderRadius: 8,
            '& fieldset': {
              borderColor: 'rgba(255, 255, 255, 0.3)',
            },
            '&:hover fieldset': {
              borderColor: 'rgba(255, 255, 255, 0.5)',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#ffffff',
            },
          },
        },
      },
    },
  },
});

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [caption, setCaption] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.size > 10 * 1024 * 1024) { // 10MB size limit
        setError("File is too large. Please upload an image under 10MB.");
        return;
      }
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setCaption("");
      setError("");
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    setIsLoading(true);
    setError("");
    setCaption("");
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Send the image to the FastAPI backend
      const response = await axios.post(
        "http://127.0.0.1:8000/generate-caption",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setCaption(response.data.caption);
    } catch (err) {
      console.error(err);
      const errorMsg = err.response?.data?.detail || "Failed to generate caption. Is the backend server running?";
      setError(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(caption);
  };

  const speakCaption = () => {
    if (caption && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel(); // Stop any previous speech
      const utterance = new SpeechSynthesisUtterance(caption);
      utterance.lang = 'en-US';
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <ThemeProvider theme={glassTheme}>
      <Container maxWidth="lg">
        <Paper 
          className="glass-card" 
          elevation={0}
          sx={{ mt: { xs: 2, md: 5 }, mb: { xs: 2, md: 5 }, p: { xs: 2, md: 4 } }}
        >
          <Typography 
            variant="h3" 
            component="h1" 
            textAlign="center" 
            gutterBottom
            sx={{ fontWeight: 700, letterSpacing: '0.5px' }}
          >
            AI Accessibility Generator
          </Typography>
          <Typography 
            variant="h6" 
            color="text.secondary" 
            textAlign="center" 
            sx={{ mb: 4 }}
          >
            Upload an image to generate detailed alt-text for your LMS
          </Typography>

          <Grid container spacing={4}>
            {/* Left Column: Image Upload */}
            <Grid item xs={12} md={6}>
              <Box
                sx={{
                  border: `2px dashed ${alpha(glassTheme.palette.text.primary, 0.4)}`,
                  borderRadius: 2,
                  height: 350,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexDirection: 'column',
                  background: alpha(glassTheme.palette.common.black, 0.1),
                  position: 'relative',
                  overflow: 'hidden',
                  p: 2,
                }}
              >
                {preview ? (
                  <img
                    src={preview}
                    alt="Uploaded preview"
                    style={{ 
                      width: '100%', 
                      height: '100%', 
                      objectFit: 'contain', 
                      position: 'absolute',
                      top: 0,
                      left: 0,
                    }}
                  />
                ) : (
                  <Box textAlign="center" color="text.secondary">
                    <ImageIcon sx={{ fontSize: 60, mb: 2 }} />
                    <Typography>Your image preview will appear here</Typography>
                  </Box>
                )}
              </Box>
              <Button
                variant="contained"
                component="label"
                startIcon={<UploadFile />}
                fullWidth
                sx={{ mt: 2, height: 56, fontSize: '1rem' }}
              >
                {file ? file.name : "Select Image"}
                <input type="file" hidden onChange={handleFileChange} accept="image/png, image/jpeg" />
              </Button>
            </Grid>
            
            {/* Right Column: Caption Output */}
            <Grid item xs={12} md={6}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleSubmit}
                disabled={isLoading || !file}
                size="large"
                fullWidth
                sx={{ 
                  height: 56, 
                  fontSize: '1rem',
                  background: 'white',
                  color: '#121212',
                  '&:hover': { background: '#f0f0f0' }
                }}
              >
                {isLoading ? <CircularProgress size={24} color="inherit" /> : "Generate Detailed Caption"}
              </Button>

              <Box sx={{ mt: 2, position: 'relative' }}>
                <TextField
                  fullWidth
                  multiline
                  rows={9}
                  variant="outlined"
                  value={isLoading ? "Generating... this may take a moment." : caption}
                  placeholder="Your detailed caption will appear here..."
                  InputProps={{
                    readOnly: true,
                  }}
                />
                <Box sx={{ position: 'absolute', top: 8, right: 8, display: 'flex', flexDirection: 'column' }}>
                  <Tooltip title="Copy to Clipboard">
                    <span>
                      <IconButton 
                        onClick={copyToClipboard} 
                        disabled={!caption || isLoading}
                        size="small"
                        sx={{ background: alpha(glassTheme.palette.common.black, 0.2), mb: 1 }}
                      >
                        <CopyAll fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                  <Tooltip title="Read Aloud">
                    <span>
                      <IconButton 
                        onClick={speakCaption} 
                        disabled={!caption || isLoading}
                        size="small"
                        sx={{ background: alpha(glassTheme.palette.common.black, 0.2) }}
                      >
                        <VolumeUp fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                </Box>
              </Box>
            </Grid>
          </Grid>
          
          {/* Error Display */}
          {error && (
            <Typography 
              color="error" 
              textAlign="center" 
              sx={{ 
                mt: 3, 
                fontWeight: 500, 
                background: alpha(glassTheme.palette.error.dark, 0.5), 
                p: 1.5, 
                borderRadius: 2 
              }}
            >
              {error}
            </Typography>
          )}
        </Paper>
      </Container>
    </ThemeProvider>
  );
}

export default App;