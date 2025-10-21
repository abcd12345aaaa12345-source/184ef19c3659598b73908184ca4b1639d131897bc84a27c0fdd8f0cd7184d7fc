import pandas as pd
import numpy as np
import openpyxl
from typing import Dict, List, Union

class ProlongationAnalyzer:
    """
    Класс для анализа коэффициентов пролонгации проектов
    """
    def __init__(self):
        self.months = [
            "Ноябрь 2022", "Декабрь 2022", "Январь 2023", "Февраль 2023", "Март 2023",
            "Апрель 2023", "Май 2023", "Июнь 2023", "Июль 2023", "Август 2023",
            "Сентябрь 2023", "Октябрь 2023", "Ноябрь 2023", "Декабрь 2023",
            "Январь 2024", "Февраль 2024"
        ]
        self.month_to_index = {m: i for i, m in enumerate(self.months)}
        self.trans_table = str.maketrans({'\xa0': '', ' ': '', ',': '.'})
    
    def clean_numeric_value(self, value: Union[str, float, int]) -> float:
        """
        Очистка и преобразование числовых значений в float
        """
        if pd.isna(value) or value in [0, '0', 'стоп', 'end', 'в ноль']:
            return 0.0
        
        return float(str(value).translate(self.trans_table))
    
    def load_and_clean_data(self, prol_path: str, fin_path: str) -> pd.DataFrame:
        """
        Загрузка и очистка исходных данных
        """
        # Загрузка данных
        df_prol = pd.read_csv(prol_path, sep=",")
        df_fin = pd.read_csv(fin_path, sep=",")
        
        # Предобработка данных по пролонгациям
        df_prol = self._preprocess_prolongations(df_prol)
        
        # Синхронизация ID между датасетами
        df_fin = self._sync_datasets_ids(df_prol, df_fin)
        
        # Объединение и очистка данных
        return self._merge_and_clean_data(df_prol, df_fin)
    
    def _preprocess_prolongations(self, df_prol: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка данных по пролонгациям
        """
        # Удаляем дубликаты, оставляя последнюю запись
        df_prol = df_prol.drop_duplicates(subset=['id'], keep='last')
        
        # Форматируем названия месяцев
        df_prol['month'] = df_prol['month'].str.strip().str.title()
        
        return df_prol
    
    def _sync_datasets_ids(self, df_prol: pd.DataFrame, df_fin: pd.DataFrame) -> pd.DataFrame:
        """
        Синхронизация ID между датасетами
        """
        prol_ids = set(df_prol['id'].unique())
        fin_ids = set(df_fin['id'].unique())
        
        # Находим ID, которые есть в финансовых данных, но отсутствуют в пролонгациях
        missing_ids = fin_ids - prol_ids
        
        if missing_ids:
            print(f"Удалено {len(missing_ids)} проектов без данных о пролонгациях")
            df_fin = df_fin[~df_fin['id'].isin(missing_ids)]
        
        return df_fin
    
    def _merge_and_clean_data(self, df_prol: pd.DataFrame, df_fin: pd.DataFrame) -> pd.DataFrame:
        """
        Объединение и очистка данных
        """
        # Объединяем данные
        df = pd.merge(df_fin, df_prol, on="id", suffixes=('_fin', '_prol'))
        
        # Удаляем ненужные колонки
        df = df.drop(["Account", "Причина дубля"], axis=1, errors='ignore')
        
        # Обрабатываем финансовые колонки
        df = self._process_financial_columns(df)
        
        # Агрегируем данные по ID
        return self._aggregate_data(df)
    
    def _process_financial_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обработка финансовых колонок
        """
        # Заполняем пропуски и заменяем специальные значения
        df[self.months] = df[self.months].fillna(0)
        
        replace_dict = {'в ноль': '0', 'стоп': '0', 'end': '0'}
        df[self.months] = df[self.months].replace(replace_dict)
        
        # Удаляем строки с оставшимися 'стоп' и 'end'
        stop_mask = df[self.months].isin(['стоп', 'end']).any(axis=1)
        df = df[~stop_mask]
        
        # Преобразуем в числовой формат
        for month in self.months:
            df[month] = df[month].apply(self.clean_numeric_value)
        
        return df
    
    def _aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегация данных по ID проекта
        """
        agg_dict = {col: 'sum' for col in self.months}
        other_cols = [col for col in df.columns if col not in self.months and col != 'id']
        agg_dict.update({col: 'first' for col in other_cols})
        
        return df.groupby('id', as_index=False).agg(agg_dict)
    
    def adjust_zero_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Корректировка нулевых значений по правилу 'в ноль'
        """
        df_adj = df.copy()
        
        for i, row in df_adj.iterrows():
            values = row[self.months].astype(float).values
            
            if not np.any(values):
                continue
            
            # Заполняем нули предыдущими ненулевыми значениями
            for j in range(1, len(self.months)):
                if values[j] == 0 and np.any(values[:j] != 0):
                    # Находим последнее ненулевое значение
                    non_zero_mask = values[:j] != 0
                    if np.any(non_zero_mask):
                        values[j] = values[:j][non_zero_mask][-1]
            
            df_adj.loc[i, self.months] = values
        
        return df_adj
    
    def calculate_prolongation_coefficients(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет коэффициентов пролонгации
        """
        results = []
        
        for month in self.months[:-2]:
            m_idx = self.month_to_index[month]
            next_m = self.months[m_idx + 1]
            next2_m = self.months[m_idx + 2]
            
            # Проекты, завершившиеся в текущем месяце
            finished = df[df['month'] == month]
            
            if finished.empty:
                results.append({
                    'месяц завершения': month,
                    'месяц пролонгации 1': next_m,
                    'месяц пролонгации 2': next2_m,
                    'coef1': np.nan,
                    'coef2': np.nan
                })
                continue
            
            # Расчет коэффициентов
            base_sum = finished[month].sum()
            first_ext = finished[next_m].sum()
            coef1 = first_ext / base_sum if base_sum != 0 else np.nan
            
            # Для второго коэффициента учитываем только проекты без пролонгации в первом месяце
            not_first = finished[finished[next_m] == finished[month]]
            base_sum_2 = not_first[month].sum()
            second_ext = not_first[next2_m].sum()
            coef2 = second_ext / base_sum_2 if base_sum_2 != 0 else np.nan
            
            results.append({
                'месяц завершения': month,
                'месяц пролонгации 1': next_m,
                'месяц пролонгации 2': next2_m,
                'coef1': coef1,
                'coef2': coef2
            })
        
        return pd.DataFrame(results)
    
    def generate_reports(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Генерация всех отчетов
        """
        # Корректируем данные
        df_adj = self.adjust_zero_values(df)
        
        # Расчет по менеджерам
        manager_results = []
        for am, subdf in df_adj.groupby('AM'):
            res = self.calculate_prolongation_coefficients(subdf)
            res['AM'] = am
            manager_results.append(res)
        
        manager_df = pd.concat(manager_results, ignore_index=True)
        
        # Расчет по отделу
        dept_df = self.calculate_prolongation_coefficients(df_adj)
        dept_df['AM'] = 'Отдел в целом'
        
        # Годовые коэффициенты
        annual = (
            pd.concat([manager_df, dept_df], ignore_index=True)
            .groupby('AM')[['coef1', 'coef2']]
            .mean()
            .reset_index()
            .rename(columns={
                'coef1': 'Годовой коэффициент (1-й мес)',
                'coef2': 'Годовой коэффициент (2-й мес)'
            })
        )
        
        return {
            'По менеджерам': manager_df,
            'Отдел в целом': dept_df,
            'Годовые коэффициенты': annual
        }
    
    def save_to_excel(self, reports: Dict[str, pd.DataFrame], filename: str = "report.xlsx"):
        """
        Сохранение отчетов в Excel с авто-подбором ширины колонок
        """
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for sheet_name, df in reports.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Автоподбор ширины колонок
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if cell.value:
                                max_length = max(max_length, len(str(cell.value)))
                        except:
                            pass
                    
                    worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)
        
        print(f"Отчет сохранен в файл: {filename}")


def main():
    """
    Основная функция выполнения анализа
    """
    analyzer = ProlongationAnalyzer()
    
    try:
        # Загрузка и очистка данных
        print("Загрузка и обработка данных...")
        df = analyzer.load_and_clean_data("prolongations.csv", "financial_data.csv")
        
        # Сохранение промежуточных данных
        df.to_csv("processed_data.csv", index=False)
        print("Промежуточные данные сохранены в processed_data.csv")
        
        # Генерация отчетов
        print("Генерация отчетов...")
        reports = analyzer.generate_reports(df)
        
        # Сохранение в Excel
        analyzer.save_to_excel(reports, "report.xlsx")
        
        # Вывод статистики
        print(f"\nСтатистика анализа:")
        print(f"- Проанализировано проектов: {len(df)}")
        print(f"- Менеджеров: {df['AM'].nunique()}")
        print(f"- Период анализа: {len(analyzer.months)} месяцев")
        
    except FileNotFoundError as e:
        print(f"Ошибка: Файл не найден - {e}")
    except Exception as e:
        print(f"Ошибка при выполнении анализа: {e}")


if __name__ == "__main__":
    main()