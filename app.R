library(shiny)
library(reticulate)
library(httr)
library(markdown)


ui <- fluidPage(
  titlePanel("AMS Risk Prediction"),
  
  sidebarLayout(
    sidebarPanel(
      numericInput("age", "Age", 18),
      selectInput("gender", "Gender", c("M" = "M", "F" = "F")),
      numericInput("temp", "Temperature (°F) [97.7 – 99.5° considered normal]", 98),
      numericInput("bp_systolic", "Systolic Blood Pressure [90 - 120 mmHg is considered healthy]", 120),
      numericInput("bp_diastolic", "Diastolic Blood Pressure [60 - 80 mmHg is considered healthy", 80),
      numericInput("spo2", "Blood Oxygen Saturation (%) [95% - 100% considered healthy]", 98),
      numericInput("pulse", "Pulse Rate (bpm) [60 - 100 bpm considered normal]", 70),
      numericInput("BVP", "Blood Volume Pulse", -0.15),
      numericInput("EDA", "Electrodermal Activity", 0.02),
      numericInput("Theta", "Theta", -0.3),
      numericInput("Alpha1", "Alpha1", 0.014),
      numericInput("Beta1", "Beta1", -0.03),
      checkboxInput("hypertension", "Hypertension", FALSE),
      checkboxInput("diabetes", "Diabetes", FALSE),
      checkboxInput("smoking", "Smoking", FALSE),
      numericInput("sym_headache", "Symptom: Headache", 0),
      numericInput("sym_gi", "Symptom: Gastrointestinal Distress", 0),
      numericInput("sym_fatigue", "Symptom: Fatigue", 0),
      numericInput("sym_dizziness", "Symptom: Dizziness", 0),
      numericInput("permanent_altitude", "Permanent Altitude (meters)", 100),
      numericInput("alt_gain_from_altitude", "Altitude Gain From (meters)", 0),
      numericInput("alt_gain_to_altitude", "Altitude Gain To (meters)", 1000),
      numericInput("ascent_day", "Ascent Day", 1),
      actionButton("send_chatbot", "Send to Chatbot"),
      
      
      
    ),
    mainPanel(
      textOutput("riskProbability"),
      textOutput("llsScore"),
      uiOutput("chatResponse")
    )
  )
)



call_python_script <- function(input_data) {
  # Call Python script and pass the necessary parameters
  print("1")
  py_run_string("import AltitudeScript")
  print("2")
  result <- py$AltitudeScript$get_langchain_response(input_data)
  print("3")
  print(result)
  return(as.character(result))  # Convert result to a string
}


server <- function(input, output) {
  use_python("/Users/srinivas/miniforge3/envs/myenv_arm64/bin/python", required = TRUE)
  source_python("/Users/srinivas/AltitudeApp/AltitudeScript.py")
  
  prediction <- reactive({
    sample_data <- data.frame(
      age = input$age,
      gender = as.character(input$gender),
      permanent_altitude = input$permanent_altitude,
      bp_systolic = input$bp_systolic,
      bp_diastolic = input$bp_diastolic,
      spo2 = input$spo2,
      pulse = input$pulse,
      hypertension = as.integer(input$hypertension),
      diabetes = as.integer(input$diabetes),
      ascent_day = input$ascent_day,
      smoking = as.integer(input$smoking),
      sym_headache = as.integer(input$sym_headache),
      sym_gi = as.integer(input$sym_gi),
      sym_fatigue = as.integer(input$sym_fatigue),
      sym_dizziness = as.integer(input$sym_dizziness),
      alt_gain_from_altitude = input$alt_gain_from_altitude,
      alt_gain_to_altitude = input$alt_gain_to_altitude,
      temp = input$temp,
      BVP = input$BVP,
      EDA = input$EDA,
      Theta = input$Theta,
      Alpha1 = input$Alpha1,
      Beta1 = input$Beta1
    )
    
    tryCatch({
      get_prediction(sample_data)
    }, error = function(e) {
      list(predicted_lls_score = NA)
    })
  })
  
  output$llsScore <- renderText({
    result <- prediction()
    if (!is.na(result$predicted_lls_score)) {
      paste("Predicted LLS score:", result$predicted_lls_score)
    } else {
      "Error in prediction"
    }
  })
  
  # Handle the event when 'send' button is clicked
  observeEvent(input$send_chatbot, {
    # Ensure that the necessary data is available
    prediction_results <- prediction()  # Access the reactive values
    print("Preparing Input data")
    
    # Prepare the data for the OpenAI API call
    input_data <- list(
      # age = input$age,
      # gender = input$gender,
      # temp = input$temp,
      # bp_systolic = input$bp_systolic,
      # bp_diastolic = input$bp_diastolic,
      # spo2 = input$spo2,
      # pulse = input$pulse,
      # hypertension = as.integer(input$hypertension),
      # diabetes = as.integer(input$diabetes),
      # ascent_day = input$ascent_day,
      # smoking = as.integer(input$smoking),
      # sym_headache = as.integer(input$sym_headache),
      # sym_gi = as.integer(input$sym_gi),
      # sym_fatigue = as.integer(input$sym_fatigue),
      # sym_dizziness = as.integer(input$sym_dizziness),
      # permanent_altitude = input$permanent_altitude,
      # alt_gain_from_altitude = input$alt_gain_from_altitude,
      # alt_gain_to_altitude = input$alt_gain_to_altitude,
      # BVP = input$BVP,
      # EDA = input$EDA,
      # AccX = input$AccX,
      # AccY = input$AccY,
      # AccZ = input$AccZ,
      
      age = input$age,
      gender = as.character(input$gender),
      permanent_altitude = input$permanent_altitude,
      bp_systolic = input$bp_systolic,
      bp_diastolic = input$bp_diastolic,
      spo2 = input$spo2,
      pulse = input$pulse,
      hypertension = as.integer(input$hypertension),
      diabetes = as.integer(input$diabetes),
      ascent_day = input$ascent_day,
      smoking = as.integer(input$smoking),
      sym_headache = as.integer(input$sym_headache),
      sym_gi = as.integer(input$sym_gi),
      sym_fatigue = as.integer(input$sym_fatigue),
      sym_dizziness = as.integer(input$sym_dizziness),
      alt_gain_from_altitude = input$alt_gain_from_altitude,
      alt_gain_to_altitude = input$alt_gain_to_altitude,
      temp = input$temp,
      BVP = input$BVP,
      EDA = input$EDA,
      Theta = input$Theta,
      Alpha1 = input$Alpha1,
      Beta1 = input$Beta1,
      lls_score = prediction_results$predicted_lls_score
    )
    
    print("Input data prepared")
    
    # Call the OpenAI API
    print("has key")
    response <- call_python_script(input_data)
    
    # Update the output with the response
    output$chatResponse <- renderUI({
      response <- call_python_script(input_data)
      HTML(response)
    })
    
  })
  
  
  
  
}

shinyApp(ui = ui, server = server)
