package com.sparrowrecsys.online.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.datamanager.Movie;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.List;

/**
 * SearchMovieService, provide search functionality for movies
 */
public class SearchMovieService extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        try {
            response.setContentType("application/json");
            response.setStatus(HttpServletResponse.SC_OK);
            response.setCharacterEncoding("UTF-8");
            response.setHeader("Access-Control-Allow-Origin", "*");

            // 从 URL 参数中获取搜索关键词
            String query = request.getParameter("query");

            if (query == null || query.trim().isEmpty()) {
                // 如果搜索关键词为空，返回空数组
                response.getWriter().println("[]");
                return;
            }

            // 使用 DataManager 获取匹配的电影列表
            List<Movie> movies = DataManager.getInstance().searchMovies(query);

            if (movies != null && !movies.isEmpty()) {
                // 将电影列表转换为 JSON 格式并返回
                /*       *ObjectMapper：
             * 这是 Jackson 库中的核心类，用于将 Java 对象和 JSON 格式之间相互转换。**/
                ObjectMapper mapper = new ObjectMapper();
                String jsonMovies = mapper.writeValueAsString(movies);
                System.out.println("Search results found for query: " + query);
                response.getWriter().println(jsonMovies);
            } else {
                // 如果没有匹配结果，返回空数组
                System.out.println("No search results found for query: " + query);
                response.getWriter().println("[]");
            }
        } catch (Exception e) {
            e.printStackTrace();
            response.getWriter().println("[]");
        }
    }
}
